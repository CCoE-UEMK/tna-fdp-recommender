import streamlit as st
import pickle
import numpy as np

# Load your saved classifier model
with open("tna_model.pkl", "rb") as f:
    classifier = pickle.load(f)

# FDP topic map for your 12 subdomains
fdp_topic_map = {
    "A1_Knowledge_Base_TNA": ["Advanced Subject Knowledge", "New Trends in Discipline", "Interdisciplinary Knowledge"],
    "A2_Cognitive_Abilities_TNA": ["Critical Thinking Workshops", "Design Thinking", "Problem Solving Strategies"],
    "A3_Creativity_TNA": ["Creative Pedagogy", "Innovation Bootcamps", "Idea Incubation"],
    "B1_Personal_Qualities_TNA": ["Emotional Intelligence", "Resilience Building", "Mindfulness in Teaching"],
    "B2_Self_Management_TNA": ["Time Management", "Stress Management", "Balancing Teaching & Research"],
    "B3_Prof_Career_Dev_TNA": ["NEP 2020 Implementation", "Career Pathways in HEI", "Skill Enhancement FDPs"],
    "C1_Professional_Conduct_TNA": ["Professional Ethics", "Academic Integrity", "Policy Compliance"],
    "C2_Research_Management_TNA": ["Research Grant Writing", "Project Management Tools", "Publication Strategies"],
    "C3_Finance_Funding_TNA": ["Funding Agencies", "Budget Management in Projects", "Utilizing Research Grants"],
    "D1_Working_with_Others_TNA": ["Collaborative Projects", "Interdisciplinary Teams", "Team Leadership"],
    "D2_Communication_TNA": ["Academic Writing", "Presentation Skills", "Effective Student Communication"],
    "D3_Engagement_Impact_TNA": ["Industry Collaborations", "Community Outreach", "Societal Impact Projects"]
}

st.title("🎯 TNA FDP Recommender")
st.write("This app predicts if an HOD has a high need for FDPs and shows the key subdomain with recommended topics.")

# Sidebar input sliders
st.sidebar.header("📝 Enter TNA Scores (out of 10)")
scores = {
    "A1_Knowledge_Base_TNA": st.sidebar.slider("Knowledge Base (A1)", 0, 10, 5),
    "A2_Cognitive_Abilities_TNA": st.sidebar.slider("Cognitive Abilities (A2)", 0, 10, 5),
    "A3_Creativity_TNA": st.sidebar.slider("Creativity (A3)", 0, 10, 5),
    "B1_Personal_Qualities_TNA": st.sidebar.slider("Personal Qualities (B1)", 0, 10, 5),
    "B2_Self_Management_TNA": st.sidebar.slider("Self Management (B2)", 0, 10, 5),
    "B3_Prof_Career_Dev_TNA": st.sidebar.slider("Professional Career Dev (B3)", 0, 10, 5),
    "C1_Professional_Conduct_TNA": st.sidebar.slider("Professional Conduct (C1)", 0, 10, 5),
    "C2_Research_Management_TNA": st.sidebar.slider("Research Management (C2)", 0, 10, 5),
    "C3_Finance_Funding_TNA": st.sidebar.slider("Finance & Funding (C3)", 0, 10, 5),
    "D1_Working_with_Others_TNA": st.sidebar.slider("Working with Others (D1)", 0, 10, 5),
    "D2_Communication_TNA": st.sidebar.slider("Communication (D2)", 0, 10, 5),
    "D3_Engagement_Impact_TNA": st.sidebar.slider("Engagement & Impact (D3)", 0, 10, 5)
}

# Prepare the feature vector
X = np.array(list(scores.values())).reshape(1, -1)

# Predict
prediction = classifier.predict(X)[0]
probability = classifier.predict_proba(X)[0][1]

# Find subdomain with highest TNA score
highest_subdomain = max(scores, key=scores.get)
fdp_topics = fdp_topic_map[highest_subdomain]

# Display results
st.subheader("🔍 Results")
if prediction == 1:
    st.success(f"High FDP Need: ✅ YES (prob: {probability:.2f})")
else:
    st.info(f"High FDP Need: 🚫 NO (prob: {probability:.2f})")

st.write(f"📈 **Key Area Needing Focus:** `{highest_subdomain}`")
st.write("🎯 **Recommended FDP Topics:**")
for topic in fdp_topics:
    st.markdown(f"- {topic}")
