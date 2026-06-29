import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from wordcloud import WordCloud
import matplotlib.pyplot as plt


# ---------------------------------------------------
# ATS Gauge Chart
# ---------------------------------------------------

def ats_gauge(score):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "ATS Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "green"},
                "steps": [
                    {"range": [0, 40], "color": "#ffb3b3"},
                    {"range": [40, 60], "color": "#ffe699"},
                    {"range": [60, 80], "color": "#c6efce"},
                    {"range": [80, 100], "color": "#66bb6a"},
                ],
            },
        )
    )

    fig.update_layout(height=350)

    return fig


# ---------------------------------------------------
# Skill Match Bar Chart
# ---------------------------------------------------

def skill_bar_chart(matched, missing):

    df = pd.DataFrame(
        {
            "Category": ["Matched Skills", "Missing Skills"],
            "Count": [len(matched), len(missing)],
        }
    )

    fig = px.bar(
        df,
        x="Category",
        y="Count",
        text="Count",
        title="Skill Comparison",
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(height=400)

    return fig


# ---------------------------------------------------
# Pie Chart
# ---------------------------------------------------

def pie_chart(matched, missing):

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Matched", "Missing"],
                values=[len(matched), len(missing)],
                hole=0.45,
            )
        ]
    )

    fig.update_layout(
        title="Matched vs Missing Skills",
        height=400,
    )

    return fig


# ---------------------------------------------------
# Keyword Frequency Chart
# ---------------------------------------------------

def keyword_chart(keyword_list):

    if len(keyword_list) == 0:
        return None

    words = [k[0] for k in keyword_list]
    counts = [k[1] for k in keyword_list]

    df = pd.DataFrame(
        {
            "Keyword": words,
            "Frequency": counts,
        }
    )

    fig = px.bar(
        df,
        x="Keyword",
        y="Frequency",
        title="Top Resume Keywords",
        text="Frequency",
    )

    fig.update_layout(height=450)

    return fig


# ---------------------------------------------------
# Word Cloud
# ---------------------------------------------------

def create_wordcloud(text):

    if not text.strip():
        return None

    wordcloud = WordCloud(
        width=900,
        height=450,
        background_color="white",
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(wordcloud)

    ax.axis("off")

    return fig


# ---------------------------------------------------
# Resume Score Card
# ---------------------------------------------------

def score_progress(score):

    if score >= 80:
        return "🟢 Excellent"

    elif score >= 60:
        return "🟡 Good"

    elif score >= 40:
        return "🟠 Average"

    else:
        return "🔴 Poor"


# ---------------------------------------------------
# Dashboard Metrics
# ---------------------------------------------------

def dashboard_metrics(score, skills, missing):

    return {
        "ATS Score": score,
        "Skills Found": len(skills),
        "Missing Skills": len(missing),
        "Resume Status": score_progress(score),
    }
