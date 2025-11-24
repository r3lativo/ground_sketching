import streamlit as st
import pandas as pd
import os
import re

# --- CONFIGURATION ---
DATA_FOLDER = "output/experiment_1_mini"
st.set_page_config(layout="wide", page_title="Data Visualization Tool")

# Custom CSS for chat bubbles
st.markdown("""
<style>
    div.stButton > button {
        text-align: left;
        height: auto;
        padding-top: 10px;
        padding-bottom: 10px;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

def get_csv_files(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return []
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]

def backfill_images(df_chunk):
    """
    Fills in missing images using the 'last seen' image for that character.
    """
    df_chunk['display_image'] = None
    character_states = {}

    for index, row in df_chunk.iterrows():
        char = row['character']
        raw_img_path = str(row['img_path']) 
        
        # Basic validation
        is_valid_image = (
            raw_img_path.lower() != 'nan' and 
            raw_img_path.strip() != '' and 
            '[NO_CHANGE]' not in row['text'] and 
            '[NO_CHANGE]' not in raw_img_path
        )

        if is_valid_image:
            character_states[char] = raw_img_path
            df_chunk.at[index, 'display_image'] = raw_img_path
        else:
            if char in character_states:
                df_chunk.at[index, 'display_image'] = character_states[char]
            else:
                df_chunk.at[index, 'display_image'] = None
                
    return df_chunk

def assign_character_styles(df):
    unique_chars = df['character'].unique()
    style_map = {}
    
    for i, char in enumerate(unique_chars):
        if i == 1: 
            style_map[char] = {'col_ratio': [1, 4], 'align': 'right'}
        else:
            style_map[char] = {'col_ratio': [4, 1], 'align': 'left'}
    return style_map

def get_image_sequence(img_path, final_prompt):
    """
    Detects if the image is part of a sequence (e.g., _seq2.png).
    Returns a list of dictionaries: [{'path': str, 'prompt': str}]
    """
    if not img_path or not os.path.exists(img_path):
        return []

    # Split prompts by $$$
    prompts = [p.strip() for p in str(final_prompt).split("$$$") if p.strip()]
    
    # Check if filename indicates a sequence (ends with _seqN.png)
    # Regex matches "_seq" followed by digits, just before the extension
    match = re.search(r"_seq(\d+)\.(png|jpg|jpeg)$", img_path)
    
    sequence_data = []

    if match:
        # It is a sequence!
        current_seq_num = int(match.group(1))
        base_name = img_path[:match.start()] # Remove _seqN.png
        extension = match.group(2)
        
        # Reconstruct 0 to N
        for i in range(current_seq_num + 1):
            step_path = f"{base_name}_seq{i}.{extension}"
            # Try to match with prompt. If more images than prompts, use empty string
            step_prompt = prompts[i] if i < len(prompts) else "???"
            
            if os.path.exists(step_path):
                sequence_data.append({'path': step_path, 'prompt': step_prompt})
    else:
        # Standard single image
        # If prompts has $$$, it means we tried multi-step but maybe only saved one file 
        # or logic didn't trigger suffix (augmenter logic: suffix only if >1 prompt)
        
        # We return just this image, associated with the FULL prompt string or the last one?
        # Let's associate it with the full prompt for clarity.
        sequence_data.append({'path': img_path, 'prompt': final_prompt})

    return sequence_data

# --- MAIN APP ---
def main():
    st.title("💬 Data Visualization Tool")

    # 1. SIDEBAR
    st.sidebar.header("Data Source")
    csv_files = get_csv_files(DATA_FOLDER)
    
    if not csv_files:
        st.error(f"No CSV files found in folder: '{DATA_FOLDER}'")
        st.stop()

    selected_file = st.sidebar.selectbox("Select Conversation File", csv_files)
    file_path = os.path.join(DATA_FOLDER, selected_file)

    # 2. LOAD & PROCESS
    try:
        df = load_data(file_path)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    chunk_ids = df['chunk_id'].unique()
    selected_chunk_id = st.sidebar.selectbox("Select Conversation Chunk", chunk_ids)

    chunk_df = df[df['chunk_id'] == selected_chunk_id].copy()
    chunk_df = chunk_df.sort_values(by='index')
    processed_df = backfill_images(chunk_df)
    char_styles = assign_character_styles(processed_df)

    # 3. STATE TRACKING
    tracker_key = f"{selected_file}_{selected_chunk_id}"
    if 'tracker_key' not in st.session_state or st.session_state.tracker_key != tracker_key:
        st.session_state.tracker_key = tracker_key
        st.session_state.selected_index = processed_df.iloc[0]['index']

    if st.session_state.selected_index not in processed_df['index'].values:
         st.session_state.selected_index = processed_df.iloc[0]['index']

    # 4. LAYOUT
    col_chat, col_viz = st.columns([1, 1]) 

    # --- LEFT: CHAT ---
    with col_chat:
        st.subheader(f"Chunk: {selected_chunk_id}")
        for idx, row in processed_df.iterrows():
            char = row['character']
            style = char_styles.get(char, {'col_ratio': [4,1], 'avatar': '❓', 'align': 'left'})
            c1, c2 = st.columns(style['col_ratio'])
            btn_col = c1 if style['align'] == 'left' else c2
            
            with btn_col:
                btn_label = f"**{char}:** {row['text']}"
                btn_type = "primary" if st.session_state.selected_index == row['index'] else "secondary"
                if st.button(btn_label, key=f"btn_{row['index']}", use_container_width=True, type=btn_type):
                    st.session_state.selected_index = row['index']
                    st.rerun()

    # --- RIGHT: VISUALIZER (UPDATED) ---
    with col_viz:
        st.subheader("Visual Context")
        
        active_row = processed_df[processed_df['index'] == st.session_state.selected_index].iloc[0]
        img_path = active_row['display_image']
        final_prompt = active_row['final_prompt']
        
        # Retrieve Sequence
        sequence = get_image_sequence(img_path, final_prompt)

        if sequence:
            if len(sequence) > 1:
                st.info(f"Multi-step Generation Detected: {len(sequence)} steps")
                
                # Create Tabs for each step
                tabs = st.tabs([f"Step {i+1}" for i in range(len(sequence))])
                
                for i, tab in enumerate(tabs):
                    step_data = sequence[i]
                    with tab:
                        st.image(step_data['path'], caption=f"Step {i+1}: {active_row['character']}", width=450)
                        st.caption(f"**Instruction:** {step_data['prompt']}")
            else:
                # Single Image Case
                st.image(sequence[0]['path'], caption=f"POV: {active_row['character']}", width=450)
        
        elif img_path:
             st.warning(f"Image file not found on disk: {img_path}")
        else:
            st.info("No image context available yet.")

        st.divider()

        # Prompt Evolution
        st.markdown("#### Prompt Evolution")
        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            st.markdown("**Initial Prompt:**")
            st.info(active_row['initial_prompt'])
            
        with p_col2:
            st.markdown("**Final Prompt:**")
            st.success(active_row['final_prompt'])

if __name__ == "__main__":
    main()
