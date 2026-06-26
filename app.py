import os
import csv
import uuid
import warnings
import shutil
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models.tongyi import ChatTongyi

warnings.filterwarnings("ignore")

# 配置
DASHSCOPE_API_KEY = ""
BASE_VECTOR_ROOT = "./vector_stores"
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

Path(BASE_VECTOR_ROOT).mkdir(parents=True, exist_ok=True)

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatTongyi(model="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY, temperature=0.1)

@st.cache_resource(show_spinner=False)
def get_emb():
    return DashScopeEmbeddings(model="text-embedding-v2")

st.set_page_config(page_title="多PDF知识库问答｜通义千问", layout="wide")
st.title("📖 多PDF文档RAG问答｜通义千问")

# ===================== 会话状态初始化（新增分片/召回参数） =====================
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "need_process" not in st.session_state:
    st.session_state.need_process = False
if "kb_name_mapping" not in st.session_state:
    st.session_state.kb_name_mapping = {}
if "selected_kb_show" not in st.session_state:
    st.session_state.selected_kb_show = "未选择知识库"
if "error_book" not in st.session_state:
    st.session_state.error_book = {}
if "add_error_idx" not in st.session_state:
    st.session_state.add_error_idx = None
if "tip_msg" not in st.session_state:
    st.session_state.tip_msg = ""
# RAG分片、召回自定义参数
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 400
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 150
if "top_k" not in st.session_state:
    st.session_state.top_k = 4

def refresh_all_kb_mapping():
    mapping = {}
    root = Path(BASE_VECTOR_ROOT)
    if root.exists():
        for dir_id in os.listdir(root):
            dir_path = root / dir_id
            index_file = dir_path / "index.faiss"
            meta_file = dir_path / "meta.txt"
            if dir_path.is_dir() and index_file.exists() and meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    show_name = f.read().strip()
                mapping[show_name] = dir_id
    st.session_state.kb_name_mapping = mapping

def handle_qa_callback():
    q = st.session_state.user_input.strip()
    if q and not st.session_state.need_process:
        st.session_state.need_process = True

def run_qa_logic(question):
    with st.spinner("AI思考、生成答案与解析..."):
        # 使用页面自定义top_k
        search_docs = st.session_state.vectordb.similarity_search(question, k=st.session_state.top_k)
        raw_context = "\n".join([i.page_content for i in search_docs])
        history_text = "\n".join([f"历史用户提问：{item['user']}\n历史AI回答：{item['ai']}\n" for item in st.session_state.chat_history])
        prompt = """
【最高优先级强制铁则，全程必须严格执行】
1. 绝对禁止单独只输出A/B/C/D任意选项字母，所有选择题作答、选项追问，都必须输出选项对应的完整文字内容；用户追问某字母含义时，必须绑定上一轮完整题干，展开该字母对应的原文。
2. 判定「文档未提及该问题」前，必须遍历全部召回参考片段，核对核心专业关键词（如AD22、原理图、后缀名、快捷键等），关键词高度重合仅语序不同的，一律视为有相关内容，严禁轻易判定未提及。
3. 多轮对话有指代性提问（如“这题C是什么”），严格绑定紧邻的上一条用户提问的题干，禁止跨其他题目匹配内容串答案。
4. 当前仅允许使用【原始参考文档】内的内容作答，禁止调用其他PDF知识库的知识点。

【强制生成答案+解析规则】
1. 输出固定分两大块：【答案】、【解析】；
2. 选择题解析要求：说明正确选项理由，逐条简要说明其余选项错在哪；
3. 快捷键/文件后缀类解析：区分同类文件、快捷键的使用场景，说明实际画图操作区别；
4. 概念判断题解析：解释该概念在AD22软件中的作用、使用场景、电路绘图意义；
5. 解析语言简洁贴合题库复习，不要拓展文档外无关内容；
6. 若文档无相关内容，仅回复：文档未提及该问题，无需生成解析。

历史对话上下文：
{history}
原始参考文档：
{raw_context}
当前用户新提问：{q}
        """
        final_prompt = prompt.format(history=history_text, raw_context=raw_context, q=question)
        res = get_llm().invoke(final_prompt)
        st.session_state.chat_history.append({"user": question, "ai": res.content})
        st.session_state.need_process = False

refresh_all_kb_mapping()
kb_show_list = list(st.session_state.kb_name_mapping.keys())

# ===================== 新增自定义参数配置面板 =====================
with st.expander("⚙️ RAG参数自定义（新建知识库生效）", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.chunk_size = st.number_input("文本分片大小 chunk_size", min_value=100, max_value=2000, value=st.session_state.chunk_size, step=50)
    with c2:
        st.session_state.chunk_overlap = st.number_input("分片重叠 chunk_overlap", min_value=20, max_value=800, value=st.session_state.chunk_overlap, step=20)
    with c3:
        st.session_state.top_k = st.number_input("检索召回条数 top_k", min_value=1, max_value=10, value=st.session_state.top_k, step=1)
    st.info("修改参数后，仅新上传构建的知识库会使用该配置；已有旧库不受影响")

# 知识库管理区（含删除按钮）
st.subheader("📚 知识库管理")
col_kb1, col_kb2, col_kb3 = st.columns([2, 1, 1])
with col_kb1:
    select_options = ["未选择知识库"] + kb_show_list
    current_selected = st.selectbox("切换已有PDF知识库", options=select_options, index=select_options.index(st.session_state.selected_kb_show), key="kb_selector")
with col_kb2:
    st.info("切换知识库会清空当前对话历史，错题按文档独立存储")
with col_kb3:
    if current_selected != "未选择知识库" and st.button("🗑️ 删除当前知识库", type="secondary"):
        storage_id = st.session_state.kb_name_mapping[current_selected]
        target_folder = Path(BASE_VECTOR_ROOT) / storage_id
        if target_folder.exists():
            shutil.rmtree(target_folder)
        if current_selected in st.session_state.error_book:
            del st.session_state.error_book[current_selected]
        st.session_state.selected_kb_show = "未选择知识库"
        st.session_state.vectordb = None
        st.session_state.chat_history = []
        st.session_state.add_error_idx = None
        st.session_state.tip_msg = f"✅ 知识库「{current_selected}」已彻底删除"
        refresh_all_kb_mapping()
        st.rerun()

if current_selected != st.session_state.selected_kb_show:
    st.session_state.selected_kb_show = current_selected
    st.session_state.chat_history = []
    st.session_state.add_error_idx = None
    st.session_state.tip_msg = ""
    if current_selected != "未选择知识库":
        storage_id = st.session_state.kb_name_mapping[current_selected]
        target_path = Path(BASE_VECTOR_ROOT) / storage_id
        st.session_state.vectordb = FAISS.load_local(str(target_path), get_emb(), allow_dangerous_deserialization=True)
        st.success(f"✅ 已切换至知识库：{current_selected}")
    else:
        st.session_state.vectordb = None
    st.rerun()

# PDF上传构建区（使用自定义分片参数）
st.divider()
upload_col1, upload_col2 = st.columns([3, 1])
with upload_col1:
    pdf_file = st.file_uploader("上传新PDF文件，生成独立知识库", type="pdf")
with upload_col2:
    build_btn = st.button("构建当前PDF知识库", key="build_pdf_btn")

if pdf_file and build_btn:
    with st.spinner("正在解析PDF并生成独立向量库..."):
        pdf_display_name = pdf_file.name.replace(".pdf", "").strip()
        storage_id = f"kb_{uuid.uuid4().hex[:16]}"
        kb_dir = Path(BASE_VECTOR_ROOT) / storage_id
        kb_dir.mkdir(parents=True, exist_ok=True)
        tmp_pdf = Path("tmp_upload.pdf")
        if tmp_pdf.exists():
            tmp_pdf.unlink()
        tmp_pdf.write_bytes(pdf_file.read())
        loader = PyPDFLoader(str(tmp_pdf))
        docs = loader.load()
        # 读取页面自定义分片参数
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap
        )
        split_docs = splitter.split_documents(docs)
        new_db = FAISS.from_documents(split_docs, get_emb())
        new_db.save_local(kb_dir)
        meta_file = kb_dir / "meta.txt"
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(pdf_display_name)
        st.session_state.vectordb = new_db
        st.session_state.selected_kb_show = pdf_display_name
        st.session_state.chat_history = []
        st.session_state.add_error_idx = None
        st.session_state.tip_msg = ""
        refresh_all_kb_mapping()
        tmp_pdf.unlink()
        st.success(f"🎉 知识库「{pdf_display_name}」构建完成，已自动切换！")
        st.rerun()

if not st.session_state.vectordb:
    st.info("📂 当前未加载任何PDF知识库，请上传PDF并构建，或在上方下拉框选择已有知识库")

st.divider()
# 展示提示信息
if st.session_state.tip_msg != "":
    st.success(st.session_state.tip_msg)
    st.session_state.tip_msg = ""

# 渲染对话历史+按钮
st.subheader("对话历史")
for idx, item in enumerate(st.session_state.chat_history):
    st.markdown(f"**用户{idx+1}：** {item['user']}")
    st.markdown(f"**AI{idx+1}：** {item['ai']}")
    if st.button("📝 加入错题本", key=f"add_err_btn_{idx}"):
        st.session_state.add_error_idx = idx
    st.divider()

# 输入框
user_input_text = st.text_input("输入提问/追问（按下Enter直接提交）", key="user_input", on_change=handle_qa_callback)

# 问答提交逻辑
if st.session_state.need_process and st.session_state.vectordb:
    run_qa_logic(st.session_state.user_input.strip())
    st.rerun()

# 功能按钮
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("提交提问", key="submit_btn"):
        q = st.session_state.user_input.strip()
        if q and st.session_state.vectordb:
            run_qa_logic(q)
            st.rerun()
with col2:
    if st.button("清空对话历史", key="clear_btn"):
        st.session_state.chat_history = []
        st.session_state.add_error_idx = None
        st.session_state.tip_msg = ""
        st.rerun()
with col3:
    if st.button("导出全部问答记录", key="export_btn"):
        if len(st.session_state.chat_history) == 0:
            st.warning("暂无问答记录可导出！")
        else:
            csv_path = Path(__file__).parent / "qa_records.csv"
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["序号", "用户提问", "AI答案与解析"])
                for idx, item in enumerate(st.session_state.chat_history, start=1):
                    writer.writerow([idx, item["user"], item["ai"]])
            st.success(f"✅ 导出成功！文件路径：{csv_path.resolve()}")
            st.rerun()

# 错题汇总UI
st.divider()
st.subheader("📝 错题汇总（仅当前知识库）")
current_kb = st.session_state.selected_kb_show
err_list = st.session_state.error_book.get(current_kb, [])
if current_kb == "未选择知识库":
    st.info("请先选择知识库再查看错题")
elif len(err_list) == 0:
    st.info("当前文档暂无错题，在对话历史点击「加入错题本」添加题目")
else:
    for e_idx, err in enumerate(err_list, 1):
        st.markdown(f"**错题{e_idx} 提问：** {err['question']}")
        st.markdown(f"**AI回答：** {err['answer']}")
        st.divider()
    err_col1, err_col2 = st.columns([1, 1])
    with err_col1:
        if st.button("清空当前知识库错题", key="clear_err"):
            st.session_state.error_book[current_kb] = []
            st.rerun()
    with err_col2:
        if st.button("导出当前知识库错题CSV", key="export_err"):
            err_csv = Path(__file__).parent / f"错题_{current_kb}.csv"
            with open(err_csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["序号", "提问", "答案与解析"])
                for e_idx, err in enumerate(err_list, start=1):
                    w.writerow([e_idx, err["question"], err["answer"]])
            st.success(f"✅ 错题导出完成：{err_csv.resolve()}")
            st.rerun()

# 处理错题添加逻辑
if st.session_state.add_error_idx is not None:
    idx = st.session_state.add_error_idx
    item = st.session_state.chat_history[idx]
    q_text = item["user"]
    a_text = item["ai"]
    kb_name = st.session_state.selected_kb_show
    if kb_name not in st.session_state.error_book:
        st.session_state.error_book[kb_name] = []
    exist = any(err["question"] == q_text for err in st.session_state.error_book[kb_name])
    if not exist:
        st.session_state.error_book[kb_name].append({"question": q_text, "answer": a_text})
        st.session_state.tip_msg = "✅ 已加入当前文档错题本！"
    else:
        st.session_state.tip_msg = "⚠️ 该题目已存在于错题本"
    st.session_state.add_error_idx = None
    st.rerun()
