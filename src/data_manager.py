# src/data_manager.py

import pandas as pd
import numpy as np
import asyncio
import logging
from pathlib import Path
import os
from typing import List, Any

logger = logging.getLogger(__name__)

class ConversationDataManager:
    """
    Manages the conversation dataset (CSV).
    Handles loading, normalization (adapter), context retrieval, and safe file writing.
    """

    def __init__(self, source_file: str, output_file: str):
        self.source_path = Path(source_file)
        self.output_path = Path(output_file)
        self.df = pd.DataFrame()
        self._write_lock = asyncio.Lock()
        self.for_later = pd.DataFrame()
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def load_and_prepare(self) -> None:
        """
        Loads CSV and initializes ALL necessary columns strictly once.
        """
        if self.output_path.exists():
            logger.info(f"[DM] Resuming from existing output: {self.output_path}")
            # Load as string to preserve formatting (e.g. "001" chunk_ids)
            self.df = pd.read_csv(self.output_path, dtype=str)
        else:
            logger.info(f"[DM] Loading fresh source: {self.source_path}")
            self.df = pd.read_csv(self.source_path)
            self._adapt_to_standard_format()
            
            # Initialize explicit columns here
            cols_to_ensure = [
                'frame_choice', 'frame_meta', 'relation', 'imagery', 
                'initial_prompt', 'final_prompt', 'img_path', 
                'extracted_triplets', 'frame_id'
            ]
            for col in cols_to_ensure:
                if col not in self.df.columns:
                    self.df[col] = pd.NA

            # Save immediately to establish schema
            self.df.to_csv(self.output_path, index=False)

        if 'm-type' in self.df.columns:

            # Keep for later rows where 'm-type' is NOT 'text'
            self.for_later = self.df[self.df['m-type'] != 'text'].copy()

            # Only work with the rows where 'm-type' is 'text'
            self.df = self.df[self.df['m-type'] == 'text'].copy()
        
        # Ensure we have a working index
        if 'index' not in self.df.columns:
            self.df['index'] = self.df.index

    async def save(self) -> None:
        """Thread-safe save to CSV."""
        async with self._write_lock:
            if not self.for_later.empty:
                df_to_save = pd.concat([self.df, self.for_later])
                df_to_save.to_csv(self.output_path, index=False)
            else:
                self.df.to_csv(self.output_path, index=False)

    async def update_cell(self, index: int, column: str, value: Any):
        """Updates a specific cell in the dataframe."""
        if isinstance(value, str):
            self.df.at[index, column] = value.replace("\n", " ")
        else:
            self.df.at[index, column] = value

    # --- SHARED HELPER (The Core Logic) ---

    def _get_frame_start_index(self, index: int, user: str, oracle: bool = False) -> int:
        """
        Finds the index where the CURRENT frame began (The last [NEW] tag).
        If no [NEW] is found, returns 0.
        """
        # 1. Slice history up to current point
        history_df = self.df.iloc[:index + 1] # Include current row to check if IT is the start

        # 2. Find rows that are [NEW] AND belong to this user
        has_new_tag = history_df['frame_choice'].astype(str).str.contains('[NEW]', regex=False, na=False)
        is_user = history_df['character'] == user
        
        # 3. Find the last valid index
        valid_start_points = has_new_tag & is_user
        last_new_idx = valid_start_points[valid_start_points].last_valid_index()
        
        # FALLBACK: If no [NEW] is found (start of file), return 0
        return int(last_new_idx) if last_new_idx is not None else 0

    # --- Data Retrieval (Simplified & Aligned) ---

    def get_prev_prompts_for_frame(self, index: int, user: str, oracle: bool = False) -> List[str]:
        """
        Get all previous prompts belonging to the current visual scene.
        """
        start_idx = self._get_frame_start_index(index, user, oracle)
        
        # Slice from start (inclusive) to current (exclusive)
        # We don't include the current row's prompt because it hasn't been generated yet!
        relevant_slice = self.df.iloc[start_idx:index]
        
        mask = (relevant_slice['character'] == user) & \
               (relevant_slice['initial_prompt'].notna()) & \
               (relevant_slice['initial_prompt'].astype(str).str.strip() != "")

        return relevant_slice.loc[mask, 'initial_prompt'].tolist()

    def get_context_for_index(self, index: int, user: str, oracle: bool = False, include_prev: bool = False) -> List[str]:
        """
        Get conversation history.
        - include_prev=False: Returns only the current scene (from last [NEW] to index).
        - include_prev=True: Returns (Previous Scene) + <scene_change> + (Current Scene).
        """
        # 1. Identify all [NEW] boundaries for this user up to current index
        # We look at history inclusive of 'index' to see if current row is a start
        history_df = self.df.iloc[:index + 1]
        
        is_user = history_df['character'] == user
        has_new = history_df['frame_choice'].astype(str).str.contains('[NEW]', regex=False, na=False)
        
        # Get indices of all rows that marked a [NEW] scene for this user
        valid_starts = history_df.index[is_user & has_new].tolist()

        # 2. Determine the "Current" Scene Start
        # If no [NEW] tags found yet, the start is 0
        curr_start_idx = valid_starts[-1] if valid_starts else 0

        # 3. Determine the "Window" Start
        start_idx = curr_start_idx
        
        if include_prev:
            if len(valid_starts) >= 2:
                # We have at least 2 scenes, so grab the one before the current one
                prev_start_idx = valid_starts[-2]
                start_idx = prev_start_idx
            else:
                # Only 1 scene exists (or none), so we fall back to the beginning of file
                start_idx = 0

        # 4. Slice the Dataframe
        # We retrieve text from start_idx up to (but not including) the current utterance
        end_idx = index
        mask = (self.df.index >= start_idx) & (self.df.index < end_idx)
        subset = self.df.loc[mask]
        
        # Convert to list of strings
        ctx_list = subset.apply(lambda row: f"{row['character']}: {row['text']}", axis=1).tolist()

        # 5. Insert Marker
        # If we included the previous scene, we must mark where the current scene actually begins.
        # We check if curr_start_idx falls strictly inside our sliced window.
        if include_prev and start_idx < curr_start_idx < end_idx:
            # We need the relative position in the list.
            # Since 'subset' contains exactly the rows in ctx_list, we count how many rows 
            # appear *before* the curr_start_idx.
            rows_before = subset[subset.index < curr_start_idx].shape[0]
            
            if 0 <= rows_before < len(ctx_list):
                 ctx_list.insert(rows_before, f"{user} <scene_change>")

        return ctx_list

    # --- Helpers ---

    def ensure_frame_ids(self, user: str) -> None:
        """
        Populates 'frame_id' column based on [NEW] tags.
        """
        mask = self.df['character'] == user
        if not mask.any(): return
        
        new_marker_mask = (
            self.df.loc[mask, 'frame_choice']
            .astype(str)
            .str.contains('[NEW]', regex=False)
            .fillna(False)
        )
        
        # Generate IDs:
        # The first [NEW] (count 1) and everything before it (count 0)
        # are grouped into Frame 1. Frame 2 only starts at the 2nd [NEW].
        # Therefore, we clip any value less than 1 to 1.
        frame_ids = new_marker_mask.cumsum().clip(lower=1)
        
        # Apply the frame id to the frame_id column
        self.df.loc[mask, 'frame_id'] = frame_ids.apply(lambda x: f"{user}_{x}")

    def get_frame_neighborhood(self, index: int, user: str) -> dict:
        """
        Retrieves context for Previous, Current, and Next frames.
        Window definition: From the first utterance of Frame X (inclusive) 
        up to the first utterance of Frame X+1 (exclusive).
        """
        curr_id_str = self.df.at[index, 'frame_id']
        if pd.isna(curr_id_str): return {}

        try:
            curr_num = int(curr_id_str.split('_')[-1])
        except (ValueError, IndexError): return {}

        current_utterance_text = self.df.at[index, 'text']
        prev_num = curr_num - 1
        next_num = curr_num + 1

        prev_id_str = f"{user}_{prev_num}" if prev_num > 0 else None
        next_id_candidate = f"{user}_{next_num}"
        has_next = (self.df['frame_id'] == next_id_candidate).any()
        next_id_str = next_id_candidate if has_next else None

        prev_text = self._get_text_window(user, prev_num)
        curr_text = self._get_text_window(user, curr_num, utterance=current_utterance_text)
        next_text = self._get_text_window(user, next_num)

        prev_frame_meta = self._get_frame_meta_info(user, prev_num)
        curr_frame_meta = self._get_frame_meta_info(user, curr_num)
        next_frame_meta = self._get_frame_meta_info(user, next_num)

        return {
            'current_frame_id': curr_id_str,
            'prev_frame_id': prev_id_str,
            'next_frame_id': next_id_str,
            'prev_text': prev_text,
            'curr_text': curr_text,
            'next_text': next_text,
            'prev_frame_meta': prev_frame_meta,
            'curr_frame_meta': curr_frame_meta,
            'next_frame_meta': next_frame_meta
        }

    def _get_mask_with_frame_num(self, user, target_num):
        """
        Get the boolean mask for where a frame id starts and ends.
        """
        if target_num <= 0: 
            return None
            
        if target_num == 1:
            # Frame 1 catches everything from the start
            start_idx = 0
        else:
            # Start is the first occurrence of this frame ID
            curr_slice = self.df[self.df['frame_id'] == f"{user}_{target_num}"]
            if curr_slice.empty: 
                return None 
            start_idx = curr_slice.index.min()

        # Find where the NEXT frame starts to define the end of THIS frame
        next_slice = self.df[self.df['frame_id'] == f"{user}_{target_num + 1}"]
        
        if not next_slice.empty:
            # End immediately before the next frame starts
            end_idx = next_slice.index.min() - 1
        else:
            # If no next frame, go to the end of the file
            end_idx = self.df.index.max()

        # Create boolean mask
        mask = (self.df.index >= start_idx) & (self.df.index <= end_idx)
        return mask

    def _get_frame_meta_info(self, user, target_num):
        """
        Constructs mask and retrieves metadata from the last row of the frame window.
        """
        mask = self._get_mask_with_frame_num(user, target_num)
        if mask is None: return None

        frame_rows = self.df.loc[mask]
        if frame_rows.empty: return None

        # Use .iloc[-1] to get the last row of this slice safely
        last_row_of_frame_id = frame_rows.iloc[-1]

        # Ensure the column exists before accessing
        if 'frame_meta' in last_row_of_frame_id:
            return last_row_of_frame_id['frame_meta']
        return None

    def _get_text_window(self, user, target_num, utterance=None):
        """
        Helper to extract text by finding global boundaries.
        """
        mask = self._get_mask_with_frame_num(user, target_num)
        if mask is None: return ""

        subset = self.df.loc[mask].copy()

        def format_row(row):
            suffix = " <---" if (utterance is not None and str(row['text']) == str(utterance)) else ""
            return f"{row['character']}: {row['text']}{suffix}"

        return "\n".join(subset.apply(format_row, axis=1).tolist())

    def _adapt_to_standard_format(self) -> None:
        """
        Standardizes the DataFrame schema. Simplified.
        """
        df = self.df
        rename_map = {'msg': 'text', 'user': 'character'}
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        self.df = df
        
    def _is_not_empty_val(self, val):
        """Safe check for non-empty, non-NA prompt strings."""
        if pd.isna(val): return False
        return str(val).strip() != ""
