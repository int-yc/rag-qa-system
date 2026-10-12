
import os
import streamlit as st

from core.rag import RAGPipeline
from config.settings import PATH_CONFIG

st.set_page_config(page_title="RAG 智能问答助手", layout="wide")
st.title("RAG 智能问答助手")
st.caption("点击左侧上传你的 txt / pdf / md 文档，或把文档放进 data 目录，即可基于文档内容提问。")

def friendly_error(e: Exception) -> str:
    msg = str(e)
    if "timed out" in msg or "Timeout" in msg or "timeout" in msg:
        return "网络连接超时，请稍后重试。若多次出现，请检查网络或代理设置。"
    if "401" in msg or "Unauthorized" in msg or "Authentication" in msg:
        return "API 密钥无效，请检查 config/settings.py 中的 api_key 是否填写正确。"
    if "404" in msg or "Not Found" in msg:
        return "模型名称（model_name）可能不正确，请检查 config/settings.py。"
    return "操作失败，请稍后重试。"

@st.cache_resource
def get_pipeline():
    return RAGPipeline()

pipeline = get_pipeline()

with st.sidebar:
    st.header("知识库管理")

    uploaded_files = st.file_uploader(
        "上传文档（点击选择，支持多选）",
        type=["txt", "pdf", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        data_dir = PATH_CONFIG["data_dir"]
        os.makedirs(data_dir, exist_ok=True)
        saved_count = 0
        for f in uploaded_files:
            save_path = os.path.join(data_dir, f.name)
            with open(save_path, "wb") as target:
                target.write(f.getbuffer())
            saved_count += 1
        st.success(f"已保存 {saved_count} 个文档，请点击下方按钮构建索引。")

    if st.button("重新构建知识库"):
        with st.spinner("正在构建知识库索引..."):
            try:
                pipeline.build()
                st.success("知识库构建完成")
            except Exception as e:
                st.error(friendly_error(e))

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入你的问题..."):

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("正在检索文档并生成回答..."):
            try:
                answer = pipeline.ask(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(friendly_error(e))
