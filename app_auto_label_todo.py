import re
import streamlit as st
import pandas as pd

# TODO 1
from underthesea import word_tokenize, sentiment


# ============================================================================
# HÀM PHÁT HIỆN SPAM
# ============================================================================
def detect_spam(text: str) -> bool:
    t = text.lower()

    # TODO 2: check link
    if re.search(r"https?://|www\.|\.com|\.vn|\.net|bit\.ly", t):
        return True

    # TODO 3: check số điện thoại
    if re.search(r"(\d[\d\.\-]{8,}\d)", t):
        return True

    # TODO 4: từ khóa spam
    spam_keywords = [
        "liên hệ", "inbox", "dm", "giá rẻ", "miễn phí",
        "zalo", "fb", "facebook", "ship", "order"
    ]
    if any(kw in t for kw in spam_keywords):
        return True

    # TODO 5: lặp ký tự
    if re.search(r"(.)\1{5,}", t):
        return True

    return False


# ============================================================================
# STREAMLIT UI
# ============================================================================
st.set_page_config(
    page_title="Lab NLP - Tự động gán nhãn",
    layout="wide"
)

st.title("Lab NLP: Tự động gán nhãn & phân tách từ")

st.markdown(
    """
Upload file CSV (cột `id`, `text`) — ứng dụng sẽ:
- Phân tách từ
- Gán nhãn cảm xúc
"""
)

# TODO 6
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("Vui lòng upload file CSV để bắt đầu.")
    st.stop()

# TODO 7
df = pd.read_csv(uploaded_file)

# TODO 8
if not {"id", "text"}.issubset(df.columns):
    st.error("File phải có cột 'id' và 'text'")
    st.stop()

st.success(f"Đã load {len(df)} dòng. Đang xử lý...")

progress = st.progress(0)

tokenized_list = []
sentiment_list = []

for i, row in df.iterrows():
    text = str(row["text"])

    # TODO 9
    tokens = word_tokenize(text, format="text")
    tokenized_list.append(tokens)

    # TODO 10
    try:
        label = sentiment(text)
    except:
        label = "neutral"

    sentiment_list.append(label)

    progress.progress((i + 1) / len(df))

# ============================================================================
# GÁN KẾT QUẢ
# ============================================================================
df["tokenized"] = tokenized_list
df["sentiment_label"] = sentiment_list

# TODO 11
df["spam"] = df["text"].apply(lambda x: detect_spam(str(x)))

# TODO 12
df["spam_label"] = df["spam"].map({True: "spam", False: "không spam"})

# TODO 13
df["spam_label_vn"] = df["spam"].map({True: "Spam", False: "Không spam"})

# TODO 14
sentiment_vn_map = {
    "positive": "Tích cực",
    "negative": "Tiêu cực",
    "neutral": "Trung lập"
}
df["sentiment_label_vn"] = df["sentiment_label"].map(sentiment_vn_map).fillna(df["sentiment_label"])

progress.empty()
st.success("Hoàn tất xử lý!")

# ============================================================================
# HIỂN THỊ
# ============================================================================
# TODO 15
st.subheader("Thống kê")

col1, col2 = st.columns(2)

with col1:
    st.write("Cảm xúc")
    st.bar_chart(df["sentiment_label_vn"].value_counts())

with col2:
    st.write("Spam")
    st.bar_chart(df["spam_label_vn"].value_counts())


# Bảng
st.subheader("Kết quả chi tiết")

display_cols = [
    "id", "text", "tokenized",
    "spam_label", "spam_label_vn",
    "sentiment_label", "sentiment_label_vn"
]

st.dataframe(df[display_cols], use_container_width=True)

# TODO 16
csv_string = df[display_cols].to_csv(index=False, encoding="utf-8-sig")
csv_bytes = csv_string.encode("utf-8-sig")

st.download_button(
    label="Tải về file kết quả (CSV)",
    data=csv_bytes,
    file_name="auto_labels_output.csv",
    mime="text/csv"
)