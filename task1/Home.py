import streamlit as st

st.set_page_config(
    page_title="FullStackGPT Home",
    page_icon="🦙"
)

st.selectbox(
    "모델을 선택해 주세요",
    (
        "GPT-3",
        "GPT-4",
    )
)

st.markdown(
    """
    # 흠...별론데?
    별로야별로야ㅐ!!!! 벼롱론야ㅕ로ㅑㅕㄴㅇ!!

    - [Document](/DocumentGPT)
    - [Private](/PrivateGPT)
    - [Quiz](/QuizGPT)
    """
)