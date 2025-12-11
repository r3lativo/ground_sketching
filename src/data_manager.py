# src/data_manager.py

import pandas as pd
import numpy as np
import asyncio
import logging
from pathlib import Path
import os
from typing import List, Optional, Tuple, Any

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
        self.for_later = pd.DataFrame
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def load_and_prepare(self) -> None:
        """
        Loads the CSV. If the output file already exists, load that (resume).
        Otherwise, load source and apply normalization.
        """
        if self.output_path.exists():
            logger.info(f"[DM] Resuming from existing output: {self.output_path}")
            # Load as string to preserve formatting (e.g. "001" chunk_ids)
            self.df = pd.read_csv(self.output_path, dtype=str)
        else:
            logger.info(f"[DM] Loading fresh source: {self.source_path}")
            self.df = pd.read_csv(self.source_path)
            self._adapt_to_standard_format()
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

    # --- Data Retrieval & Updates ---

    def get_user_groups(self, user: str, chunk_col: str):
        """
        Returns an iterator of (chunk_id, chunk_df) for a specific user.
        Usage: for chunk_id, chunk_df in data_mgr.get_user_groups(...)
        """
        if chunk_col not in self.df.columns:
            logger.warning(f"[DM] Column {chunk_col} not found. Defaulting to 'chunk_id'.")
            chunk_col = 'chunk_id'
            
        return self.df.groupby(chunk_col)

    async def update_cell(self, index: int, column: str, value: Any):
        """Updates a specific cell in the dataframe."""
        if isinstance(value, str):
            self.df.at[index, column] = value.replace("\n", " ")
        else:
            self.df.at[index, column] = value

    def set_start_idx(self, index: int, user: str, oracle: bool = False) -> int:
        """Set start index based on context"""
        start_idx = 0

        # 2. Determine Start Index (Shared Ground)
        if not oracle:
            context_start_col = f"ctx_start_idx_{user}"
            if context_start_col in self.df.columns:
                try:
                    val = self.df.at[index, context_start_col]
                    start_idx = int(float(val)) if pd.notna(val) else 0
                except:
                    start_idx = 0
        
        self.start_idx = max(0, start_idx)
        return self.start_idx

    def get_start_idx(self) -> int:
        return self.start_idx

    def get_prev_prompts_for_frame(self, index: int, user: str, oracle: bool = False) -> List[str]:
        """
        Get all previous prompts specifically for the CURRENT image generation cycle.
        Finds the last [NEW] frame specifically associated with the requested user.
        """
        # 1. Slice history up to the current index (exclusive)
        history_df = self.df.iloc[:index]

        # 2. Find Visual Start (Last [NEW] *for this user*)
        # We need two conditions:
        # A. The row has the '[NEW]' tag
        has_new_tag = history_df['frame_choice'].astype(str).str.contains('[NEW]', regex=False, na=False)
        
        # B. The row belongs to the specific user (Critical fix)
        is_user = history_df['character'] == user

        # Combine conditions: Find rows that are NEW AND belong to USER
        valid_start_points = has_new_tag & is_user
        
        # Find the index label of the *last* time this specific user started a new frame
        last_new_idx = valid_start_points[valid_start_points].last_valid_index()
        
        # If no [NEW] is found for this user, we start from the beginning (0)
        start_idx = int(last_new_idx) if last_new_idx is not None else 0

        # 3. Select and Filter Range
        # We slice from start_idx (inclusive) to current index (exclusive)
        relevant_slice = self.df.iloc[start_idx:index]
        
        # Create boolean mask for valid prompts
        # condition: (Character matches) AND (Prompt is not empty/NaN)
        # Note: We re-check character here to ensure we only get this user's prompts 
        # within that time range (ignoring intervening prompts from other users).
        mask = (relevant_slice['character'] == user) & \
               (relevant_slice['initial_prompt'].notna()) & \
               (relevant_slice['initial_prompt'].astype(str).str.strip() != "")

        # 4. Extract and return list
        return relevant_slice.loc[mask, 'initial_prompt'].tolist()

    def get_img_for_index(self, index: int) -> str:
        """Get the image path if it exists"""
        full_imgs = self.df['img_path']

        if self._is_not_empty_val(full_imgs.index):
            if os.path.exists(full_imgs.index):
                return full_imgs.index
            else:
                logger.warning(f"[DM] '{full_imgs.index}' file does not exist.")
        return None

    def get_context_for_index(self, index: int, user: str) -> List[str]:
        """
        Retrieves context and inserts a <moved> where the user's actual visual scene began.
        """
        # 1. Prepare formatted strings
        full_formatted = self.df[['character', 'text']].agg(': '.join, axis=1)
        
        # 2. Determine Start Index (Shared Ground)
        start_idx = self.get_start_idx()

        # 3. Determine End Index (Exclusive)
        end_idx = index
        
        # 4. Get the basic context slice
        mask = (full_formatted.index >= start_idx) & (full_formatted.index < end_idx)
        ctx_list = full_formatted.loc[mask].tolist()
        
        # --- Marker Logic ---
        # Find where the user's *current* chunk began
        chunk_col = f"chunk_{user}"
        if chunk_col in self.df.columns:
            current_chunk_val = self.df.at[index, chunk_col]
            
            # Find the first index where this chunk value appears
            # Walk backwards from 'index' until value changes
            scene_start_idx = index
            for i in range(index - 1, -1, -1):
                if self.df.at[i, chunk_col] != current_chunk_val:
                    break
                scene_start_idx = i
            
            # If the scene start is inside our current context window...
            if start_idx < scene_start_idx < end_idx:
                # Calculate relative position to insert
                # indices in ctx_list correspond to range(start_idx, end_idx)
                # if continuous.
                insert_pos = scene_start_idx - start_idx
                if 0 <= insert_pos < len(ctx_list):
                    ctx_list.insert(insert_pos, f"{user} <moved>")

        return ctx_list

    # --- Adapter Logic (Private) ---

    def _adapt_to_standard_format(self) -> None:
        """
        Standardizes the DataFrame schema.
        - Renames 'msg'->'text', 'user'->'character'
        - Generates 'chunk_id' if missing
        - Calculates context start indices ('ctx_start_idx_X')
        """
        df = self.df
        
        # 1. Column Mapping
        rename_map = {'msg': 'text', 'user': 'character'}
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        # 2. Chunk ID Padding
        if 'chunk_id' in df.columns:
            # Clean float-like strings ("1.0") -> "001"
            df['chunk_id'] = df['chunk_id'].astype(str).str.replace(r'\.0$', '', regex=True)
            df['chunk_id'] = df['chunk_id'].apply(lambda x: x.zfill(3))

        # 3. Dynamic Multi-View Logic (inst columns)
        inst_columns = [c for c in df.columns if c.endswith('_inst')]
        chunk_cols_map = {} 
        
        if inst_columns:
            logger.info(f"[DM] Detected Multi-View columns: {inst_columns}")
            
            # Create sub-chunk columns (e.g., chunk_A)
            for col in inst_columns:
                prefix = col.replace('_inst', '')
                chunk_col_name = f"chunk_{prefix}"
                chunk_cols_map[prefix] = chunk_col_name
                
                loc_series = df[col].fillna("UNKNOWN_LOC")
                condition = loc_series != loc_series.shift()
                df[chunk_col_name] = condition.cumsum().fillna(0).astype(int).apply(lambda x: f"{x:03d}")

            # Calculate Start Indices
            # (Logic copied from original augmenter.py for compatibility)
            prefixes = list(chunk_cols_map.keys())
            if len(prefixes) >= 2:
                start_index_helpers = {}
                for prefix in prefixes:
                    col_chunk = chunk_cols_map[prefix]
                    mask_changed = df[col_chunk] != df[col_chunk].shift()
                    helper = pd.Series(np.nan, index=df.index)
                    helper[mask_changed] = df.index[mask_changed]
                    start_index_helpers[prefix] = helper.ffill().fillna(0).astype(int)

                for i, prefix_A in enumerate(prefixes):
                    prefix_B = prefixes[(i + 1) % len(prefixes)]
                    col_chunk_A = chunk_cols_map[prefix_A]
                    target_col_name = f"ctx_start_idx_{prefix_A}"
                    
                    mask_A_changed = df[col_chunk_A] != df[col_chunk_A].shift()
                    df[target_col_name] = np.nan
                    
                    helper_B = start_index_helpers[prefix_B]
                    df.loc[mask_A_changed, target_col_name] = helper_B.loc[mask_A_changed]
                    df[target_col_name] = df[target_col_name].ffill()
                    
                    if not df.empty:
                        df.at[0, target_col_name] = helper_B.at[0]
                        df[target_col_name] = df[target_col_name].ffill()
                    
                    df[target_col_name] = df[target_col_name].astype(int)
        
        # 4. Initialize columns if missing
        for col in ['frame_choice', 'frame_meta', 'relation', 'imagery', 'initial_prompt', 'final_prompt', 'img_path']:
            if col not in df.columns:
                df[col] = pd.NA

        self.df = df

    def _is_not_empty_val(self, val):
        """Safe check for non-empty, non-NA prompt strings."""
        if pd.isna(val):
            return False
        return str(val).strip() != ""
