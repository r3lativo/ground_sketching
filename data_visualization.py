import streamlit as st
import pandas as pd
import os
import re

# --- CONFIGURATION ---
DATA_FOLDER = "output/"  # Or "output/mock"
st.set_page_config(layout="wide", page_title="Ground Sketching Viz")

# --- CSS STYLING ---
# We use CSS to highlight the "Context" rows differently from standard rows
st.markdown("""
<style>
    div.stButton > button {
        text-align: left;
        height: auto;
        padding-top: 5px;
        padding-bottom: 5px;
        white-space: pre-wrap;
    }
    /* Highlight for Context Rows (Custom class if we could inject it, 
       but for now we rely on visual markers in text) */
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Ensure index column exists and is sorted
    if 'index' not in df.columns:
        df['index'] = df.index
    return df.sort_values(by='index')

def get_csv_files(folder_path):
    if not os.path.exists(folder_path):
        return []
    files = []
    for root, dirs, filenames in os.walk(folder_path):
        for f in filenames:
            if f.endswith('.csv'):
                files.append(os.path.join(root, f))
    return files

def get_image_sequence(img_path, final_prompt):
    """Detects multi-step sequences."""
    if pd.isna(img_path) or not img_path:
        return []

    prompts = [p.strip() for p in str(final_prompt).split("$$$") if p.strip()] if pd.notna(final_prompt) else []
    match = re.search(r"_seq(\d+)\.(png|jpg|jpeg)$", img_path)
    
    sequence_data = []
    if match:
        current_seq_num = int(match.group(1))
        base_name = img_path[:match.start()]
        extension = match.group(2)
        for i in range(current_seq_num + 1):
            step_path = f"{base_name}_seq{i}.{extension}"
            step_prompt = prompts[i] if i < len(prompts) else "???"
            if os.path.exists(step_path):
                sequence_data.append({'path': step_path, 'prompt': step_prompt})
    else:
        sequence_data.append({'path': img_path, 'prompt': str(final_prompt) if pd.notna(final_prompt) else ""})
    return sequence_data

def assign_character_styles(df):
    unique_chars = [c for c in df['character'].unique() if pd.notna(c)]
    style_map = {}
    for i, char in enumerate(unique_chars):
        if i % 2 != 0: 
            style_map[char] = {'col_ratio': [1, 4], 'align': 'right'}
        else:
            style_map[char] = {'col_ratio': [4, 1], 'align': 'left'}
    return style_map

# --- MAIN APP ---
def main():
    st.title("💬 Conversation Context Visualizer")

    # 1. SIDEBAR: File Selection Only
    st.sidebar.header("Data Source")
    all_files = get_csv_files(DATA_FOLDER)
    if not all_files:
         st.error(f"No CSV files found in {DATA_FOLDER}")
         st.stop()
         
    file_map = {os.path.basename(f): f for f in all_files}
    selected_filename = st.sidebar.selectbox("Select File", list(file_map.keys()))
    file_path = file_map[selected_filename]

    # 2. LOAD DATA
    try:
        df = load_data(file_path)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    char_styles = assign_character_styles(df)

    # 3. STATE MANAGEMENT
    tracker_key = f"{selected_filename}"
    if 'tracker_key' not in st.session_state or st.session_state.tracker_key != tracker_key:
        st.session_state.tracker_key = tracker_key
        # Default to first row
        st.session_state.selected_index = df.iloc[0]['index']

    # 4. CONTEXT CALCULATION (For the Selected Row)
    # Get the row the user clicked on
    active_row = df[df['index'] == st.session_state.selected_index].iloc[0]
    
    # Who is speaking?
    active_char = active_row['character']
    
    # What is their context start?
    ctx_col = f"ctx_start_idx_{active_char}"
    
    context_indices = []
    if ctx_col in df.columns:
        start_idx = active_row[ctx_col]
        if pd.notna(start_idx):
            # Context is [Start, Current] (inclusive of current row usually in this logic)
            context_indices = list(range(int(start_idx), int(active_row['index']) + 1))
    else:
        # Fallback if column missing
        context_indices = [active_row['index']]

    # 5. LAYOUT
    col_chat, col_viz = st.columns([1, 1]) 

    # --- LEFT: CHAT HISTORY ---
    with col_chat:
        st.subheader("Conversation")
        
        # We iterate through the whole DF
        for idx, row in df.iterrows():
            char = row['character']
            style = char_styles.get(char, {'col_ratio': [4,1], 'align': 'left'})
            
            c1, c2 = st.columns(style['col_ratio'])
            btn_col = c1 if style['align'] == 'left' else c2
            
            with btn_col:
                row_idx = row['index']
                
                # Determine Visual State
                is_selected = (row_idx == st.session_state.selected_index)
                is_in_context = (row_idx in context_indices)
                
                # Visual Markers
                prefix = ""
                if is_selected:
                    btn_type = "primary"
                    prefix = "⬛ " # Marker for "You are here"
                elif is_in_context:
                    btn_type = "secondary" # Streamlit doesn't support tertiary colors well
                    prefix = "👁️ " # Marker for "Included in Context"
                else:
                    btn_type = "secondary"
                    prefix = ""

                # Construct Label
                txt = str(row['text'])
                # Optional: Add line break for clearer reading
                label = f"{prefix}**{char}:** {txt}"
                
                # Render Button
                if st.button(label, key=f"btn_{row_idx}", width='stretch', type=btn_type):
                    st.session_state.selected_index = row_idx
                    st.rerun()

    # --- RIGHT: INSPECTOR ---
    with col_viz:
        st.subheader(f"Inspector (Index {active_row['index']})")
        
        # 1. Context Explanation
        st.info(f"**Perspective:** {active_char} | **Context Window:** Index {context_indices[0]} ➡ {context_indices[-1]}")
        
        with st.expander("📄 See Exact Context Text", expanded=False):
            # Reconstruct the exact text block passed to the model
            ctx_df = df[df['index'].isin(context_indices)]
            for _, ctx_row in ctx_df.iterrows():
                if ctx_row['index'] == active_row['index']:
                    st.markdown(f"**{ctx_row['character']}: {ctx_row['text']}** (Target)")
                else:
                    st.text(f"{ctx_row['character']}: {ctx_row['text']}")

        # 2. Image Visualization
        img_path = active_row.get('img_path')
        if pd.notna(img_path) and os.path.exists(str(img_path)):
             # Sequence Logic
            sequence = get_image_sequence(img_path, active_row.get('final_prompt'))
            
            if sequence:
                if len(sequence) > 1:
                    tabs = st.tabs([f"Step {i+1}" for i in range(len(sequence))])
                    for i, tab in enumerate(tabs):
                        with tab:
                            st.image(sequence[i]['path'], caption=sequence[i]['prompt'], width='stretch')
                else:
                    st.image(sequence[0]['path'], caption="Result", width='stretch')
            else:
                st.warning("Image path exists but file read failed.")
        else:
            if pd.isna(img_path):
                st.info("No image generated for this line.")
            else:
                st.warning(f"File missing: {img_path}")

        st.divider()

        # 3. Prompts
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Initial Prompt**")
            val = active_row.get('initial_prompt')
            if pd.notna(val) and val != "":
                st.caption(val)
            else:
                st.text("-")
        with c2:
            st.markdown("**Final Prompt**")
            val = active_row.get('final_prompt')
            if pd.notna(val) and val != "":
                st.caption(val)
            else:
                st.text("-")

if __name__ == "__main__":
    main()
