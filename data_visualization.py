import streamlit as st
import pandas as pd
import os
import re
import glob

# --- CONFIGURATION ---
DATA_FOLDER = "output" 

st.set_page_config(layout="wide", page_title="Ground Sketching Viz")

# --- CSS STYLING ---
st.markdown("""
<style>
    /* General Button Styling to look more like bubbles */
    div.stButton > button {
        text-align: left;
        height: auto;
        padding: 10px;
        white-space: pre-wrap;
        width: 100%;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        line-height: 1.4;
    }
    
    /* We can't strictly target Left vs Right buttons with pure CSS in Streamlit 
       without adding custom classes (which is hard in standard st). 
       We rely on the column layout for positioning. */
       
    .img-caption {
        font-size: 0.9em;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_all_csv_files(root_folder):
    """Recursively find all .csv files."""
    csv_files = []
    if not os.path.exists(root_folder):
        st.error(f"Folder '{root_folder}' not found. Check DATA_FOLDER.")
        return []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return sorted(csv_files)

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'index' not in df.columns:
            df['index'] = df.index
        return df.sort_values(by='index')
    except Exception as e:
        return pd.DataFrame()

def get_image_sequence(img_path_in_csv, final_prompt_text):
    """
    Returns [{'path':..., 'prompt':..., 'seq_id':...}, ...]
    """
    if pd.isna(img_path_in_csv):
        return []

    # 1. Resolve Path
    if os.path.exists(img_path_in_csv):
        valid_path = img_path_in_csv
    elif os.path.exists(os.path.join(DATA_FOLDER, img_path_in_csv)):
        valid_path = os.path.join(DATA_FOLDER, img_path_in_csv)
    else:
        return []

    directory = os.path.dirname(valid_path)
    filename = os.path.basename(valid_path)
    
    # 2. Extract Prefix
    match = re.match(r"(.*)_seq(\d+)(\..+)$", filename)
    
    # Logic: if current file is NOT a sequence (no _seqN), just return it
    if not match:
        return [{'path': valid_path, 'prompt': final_prompt_text, 'seq_id': 'Final', 'is_current': True}]
    
    prefix = match.group(1)
    extension = match.group(3)
    current_seq_num = int(match.group(2))
    
    # 3. Find siblings
    search_pattern = os.path.join(directory, f"{prefix}_seq*{extension}")
    found_files = glob.glob(search_pattern)
    
    # 4. Sort
    def extract_seq_num(path):
        fname = os.path.basename(path)
        m = re.match(r".*_seq(\d+)\..+", fname)
        return int(m.group(1)) if m else 0

    sorted_files = sorted(found_files, key=extract_seq_num)
    
    # 5. Split Prompts
    prompt_parts = final_prompt_text.split('$$$') if pd.notna(final_prompt_text) else []
    
    sequence = []
    for i, file_p in enumerate(sorted_files):
        p_text = prompt_parts[i].strip() if i < len(prompt_parts) else ""
        s_num = extract_seq_num(file_p)
        
        sequence.append({
            'path': file_p,
            'prompt': p_text,
            'seq_id': f"Seq {s_num}",
            'is_current': (s_num == current_seq_num) # Mark if this is the one in the CSV row
        })
        
    return sequence

# --- MAIN APP ---

def main():
    st.sidebar.title("Configuration")
    
    csv_files = get_all_csv_files(DATA_FOLDER)
    if not csv_files:
        st.warning("No CSV files found.")
        return

    selected_file = st.sidebar.selectbox(
        "Select Conversation", 
        csv_files, 
        format_func=lambda x: x.replace(DATA_FOLDER + os.sep, "")
    )
    
    df = load_data(selected_file)
    if df.empty: return

    try:
        chat_df = df[df['m-type'] == 'text'].copy()
    except:
        chat_df = df
    if chat_df.empty:
        st.warning("No text messages found.")
        return

    if 'selected_idx' not in st.session_state:
        st.session_state.selected_idx = chat_df.iloc[0]['index']
    
    # Validate selection
    if st.session_state.selected_idx not in df['index'].values:
        st.session_state.selected_idx = chat_df.iloc[0]['index']

    # --- LAYOUT: 60% Chat, 40% Details ---
    col_chat, col_details = st.columns([5, 5])

    # --- LEFT: CHAT ---
    with col_chat:
        st.subheader("Chat")
        
        # Calculate Context
        current_row = df[df['index'] == st.session_state.selected_idx]
        ctx_indices = set()
        if not current_row.empty:
            c_row = current_row.iloc[0]
            start_a = c_row.get('ctx_start_idx_A', -1)
            start_b = c_row.get('ctx_start_idx_B', -1)
            char = c_row.get('character', '')
            
            # Simple context logic
            start_idx = start_a if char == 'A' else (start_b if char == 'B' else max(start_a, start_b))
            if start_idx >= 0:
                ctx_indices = set(df[(df['index'] >= start_idx) & (df['index'] <= st.session_state.selected_idx)]['index'].values)

        # Build list of characters
        characters = list(chat_df['character'].values)

        # Render Messages
        for _, row in chat_df.iterrows():
            idx = row['index']
            character = row.get('character', '?')
            text = row.get('text', '')
            
            is_selected = (idx == st.session_state.selected_idx)
            is_ctx = (idx in ctx_indices)
            
            # Visual Marker
            eye = "👁️ " if is_ctx else ""
            label = f"{eye}{character}: {text}"
            
            # Alignment Logic
            if character == characters[0]:
                # Empty col then Content col
                c_spacer, c_btn = st.columns([1, 3]) 
                with c_btn:
                    # Highlight selected
                    type_ = "primary" if is_selected else "secondary"
                    if st.button(label, key=f"msg_{idx}", type=type_, use_container_width=True):
                        st.session_state.selected_idx = idx
                        st.rerun()
            else:
                # Content col then Empty col
                c_btn, c_spacer = st.columns([3, 1])
                with c_btn:
                    type_ = "primary" if is_selected else "secondary"
                    if st.button(label, key=f"msg_{idx}", type=type_, use_container_width=True):
                        st.session_state.selected_idx = idx
                        st.rerun()

        try:
            st.divider()
            st.markdown("#### Summary / Q&A")
            qa_df = df[(df['m-type'].isin(['Question', 'Answer'])) | (df['m-type'].isna())]
            qa_df = qa_df[qa_df['m-type'] != 'text']
            for _, r in qa_df.iterrows():
                mtype = r.get('m-type', 'Short Answer')
                st.text(f"[{mtype}] {r.get('text', '')}")
        except:
            pass

    # --- RIGHT: DETAILS ---
    with col_details:
        if current_row.empty:
            st.info("Select a message.")
        else:
            row = current_row.iloc[0]
            st.subheader(f"Utterance {int(row['index'])}")
            
            img_path = row.get('img_path')
            final_prompt = row.get('final_prompt', "")
            
            has_image = False
            if pd.notna(img_path) and str(img_path).lower() != 'nan' and str(img_path).strip() != '':
                sequence = get_image_sequence(img_path, final_prompt)
                
                if sequence:
                    has_image = True
                    
                    # 1. Determine Default Selection
                    # Find which item in the sequence matches the current row's img_path
                    default_idx = 0
                    for i, item in enumerate(sequence):
                        if item['is_current']:
                            default_idx = i
                            break
                    
                    # 2. Render Selector (Radio or Tabs)
                    # We use radio for better programmatic control of "index"
                    # Key must be unique per row to allow resetting the default when switching rows
                    if len(sequence) > 1:
                        selected_seq_label = st.radio(
                            "Sequence Step", 
                            [s['seq_id'] for s in sequence], 
                            index=default_idx, 
                            horizontal=True,
                            key=f"seq_sel_{int(row['index'])}" 
                        )
                        # Find the data object for the selected label
                        selected_item = next(s for s in sequence if s['seq_id'] == selected_seq_label)
                    else:
                        selected_item = sequence[0]

                    # 3. Show Image (Restricted Width)
                    st.image(selected_item['path'], width=500)
                    
                    if selected_item['prompt']:
                        st.info(f"**Prompt:** {selected_item['prompt']}")
                    else:
                        st.caption("No sub-prompt for this step.")
                        
                    st.caption(f"File: `{os.path.basename(selected_item['path'])}`")
                    
                else:
                    st.warning(f"File not found: `{img_path}`")
            
            if not has_image:
                st.info("No image available.")

            st.divider()
            
            # Metadata Grid
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Frame**\n\n{row.get('frame_choice', '-')}")
            c2.markdown(f"**Imagery**\n\n{row.get('imagery_utterance', '-')}")
            c3.markdown(f"**Meta**\n\n{row.get('meta_info', '-')}")
            
            # Prompts
            st.divider()
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Initial Prompt**")
                val = row.get('initial_prompt')
                if pd.notna(val) and val != "":
                    st.caption(val)
                else:
                    st.text("-")
            with cc2:
                st.markdown("**Final Prompt**")
                val = row.get('final_prompt')
                if pd.notna(val) and val != "":
                    st.caption(val)
                else:
                    st.text("-")

if __name__ == "__main__":
    main()
