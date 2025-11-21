import streamlit as st
import pandas as pd
import os

# --- CONFIGURATION ---
DATA_FOLDER = "output/experiment_2"
st.set_page_config(layout="wide", page_title="Data Visualization Tool")

# Custom CSS for chat bubbles
st.markdown("""
<style>
    div.stButton > button {
        text-align: left;
        height: auto;
        padding-top: 10px;
        padding-bottom: 10px;
        white-space: pre-wrap; /* Ensures long text wraps nicely */
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

def get_csv_files(folder_path):
    """Scans the directory for .csv files."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return []
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]

def backfill_images(df_chunk):
    """
    Fills in missing images using the 'last seen' image for that SPECIFIC character.
    """
    df_chunk['display_image'] = None
    character_states = {}

    for index, row in df_chunk.iterrows():
        char = row['character']
        raw_img_path = str(row['img_path']) 
        
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
    """
    Dynamically determines who goes on Left vs Right.
    Returns a dictionary mapping character names to layout configs.
    """
    unique_chars = df['character'].unique()
    
    # Default Avatars to cycle through
    avatars = ["🦊", "🦉", "🤖", "👽"]
    
    style_map = {}
    
    # Logic: 
    # 1st character found -> LEFT alignment
    # 2nd character found -> RIGHT alignment
    # 3rd+ character found -> LEFT alignment (Group chat style)
    
    for i, char in enumerate(unique_chars):
        avatar = avatars[i % len(avatars)]
        
        if i == 1: # The second character goes to the RIGHT
            style_map[char] = {
                'col_ratio': [1, 4], # Spacer left, Content right
                'avatar': avatar,
                'align': 'right'
            }
        else: # Everyone else goes to the LEFT
            style_map[char] = {
                'col_ratio': [4, 1], # Content left, Spacer right
                'avatar': avatar,
                'align': 'left'
            }
            
    return style_map

# --- MAIN APP ---
def main():
    st.title("💬 Data Visualization Tool")

    # 1. SIDEBAR: FILE SELECTION
    st.sidebar.header("Data Source")
    csv_files = get_csv_files(DATA_FOLDER)
    
    if not csv_files:
        st.error(f"No CSV files found in folder: '{DATA_FOLDER}'")
        st.stop()

    selected_file = st.sidebar.selectbox("Select Conversation File", csv_files)
    file_path = os.path.join(DATA_FOLDER, selected_file)

    # 2. LOAD DATA
    try:
        df = load_data(file_path)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # 3. SIDEBAR: CHUNK SELECTION
    chunk_ids = df['chunk_id'].unique()
    selected_chunk_id = st.sidebar.selectbox("Select Conversation Chunk", chunk_ids)

    # 4. PROCESS DATA
    chunk_df = df[df['chunk_id'] == selected_chunk_id].copy()
    chunk_df = chunk_df.sort_values(by='index')
    processed_df = backfill_images(chunk_df)
    
    # Generate Dynamic Styles for this specific file's characters
    char_styles = assign_character_styles(processed_df)

    # --- STATE MANAGEMENT (Prevents Index Error on switch) ---
    # We track both File AND Chunk changes now
    tracker_key = f"{selected_file}_{selected_chunk_id}"
    
    if 'tracker_key' not in st.session_state:
        st.session_state.tracker_key = tracker_key
        st.session_state.selected_index = processed_df.iloc[0]['index']
    
    if st.session_state.tracker_key != tracker_key:
        st.session_state.tracker_key = tracker_key
        st.session_state.selected_index = processed_df.iloc[0]['index']

    # Safety fallback
    if st.session_state.selected_index not in processed_df['index'].values:
         st.session_state.selected_index = processed_df.iloc[0]['index']

    # 5. LAYOUT
    col_chat, col_viz = st.columns([1, 1]) 

    # --- LEFT COLUMN: DYNAMIC CHAT ---
    with col_chat:
        st.subheader(f"Chunk: {selected_chunk_id}")
        
        for idx, row in processed_df.iterrows():
            char = row['character']
            style = char_styles.get(char, {'col_ratio': [4,1], 'avatar': '❓'}) # Fallback
            
            # Dynamic Columns based on character style
            c1, c2 = st.columns(style['col_ratio'])
            
            # Determine where to put the button based on alignment
            if style['align'] == 'left':
                btn_col = c1
            else:
                btn_col = c2
            
            with btn_col:
                btn_label = f"{style['avatar']} {char}: {row['text']}"
                
                is_selected = (st.session_state.selected_index == row['index'])
                btn_type = "primary" if is_selected else "secondary"
                
                if st.button(btn_label, key=f"btn_{row['index']}", use_container_width=True, type=btn_type):
                    st.session_state.selected_index = row['index']
                    st.rerun()

    # --- RIGHT COLUMN: VISUALIZER ---
    with col_viz:
        st.subheader("Visual Context")
        
        active_row = processed_df[processed_df['index'] == st.session_state.selected_index].iloc[0]
        img_path = active_row['display_image']
        
        # Image Display
        if img_path and os.path.exists(img_path):
            st.image(img_path, caption=f"POV: {active_row['character']}", width=450)
        elif img_path:
             st.warning(f"Image not found: {img_path}")
        else:
            st.info("No image context available yet.")

        st.divider()

        # Prompts
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
