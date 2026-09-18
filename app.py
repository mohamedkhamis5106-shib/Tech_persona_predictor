import random

import pandas as pd
import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

random.seed(42)

st.set_page_config(
    page_title="PersonaLab | Tech Persona Predictor / متنبئ الشخصية التقنية",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- Model setup ----------
def build_persona_model():
    focus_options = [
        "Solving puzzles & logic",
        "Numbers & patterns",
        "Building smart systems",
        "Design & creativity",
        "Exploring & breaking things",
    ]
    drink_options = ["Coffee", "Tea"]
    schedule_options = ["Night owl", "Early bird"]
    approach_options = [
        "Deep focus alone",
        "Brainstorming with others",
        "Trial & error experiments",
        "Careful planning",
    ]
    energy_options = [
        "Curious & analytical",
        "Rebellious & curious",
        "Visionary & ambitious",
        "Expressive & visual",
    ]

    persona_by_focus = {
        "Solving puzzles & logic": "Hacker",
        "Numbers & patterns": "Data Scientist",
        "Building smart systems": "AI Engineer",
        "Design & creativity": "Designer",
        "Exploring & breaking things": "Hacker",
    }

    rows = []
    for _ in range(150):
        focus = random.choice(focus_options)
        drink = random.choice(drink_options)
        schedule = random.choice(schedule_options)
        approach = random.choice(approach_options)
        energy = random.choice(energy_options)

        persona = persona_by_focus.get(focus, "Data Scientist")

        if energy == "Rebellious & curious" and schedule == "Night owl":
            persona = "Hacker"
        elif energy == "Visionary & ambitious" and focus == "Building smart systems":
            persona = "AI Engineer"
        elif energy == "Expressive & visual" and approach == "Trial & error experiments":
            persona = "Designer"
        elif energy == "Curious & analytical" and focus == "Numbers & patterns":
            persona = "Data Scientist"
        elif approach == "Deep focus alone" and focus == "Solving puzzles & logic":
            persona = "Hacker"
        elif approach == "Careful planning" and focus == "Numbers & patterns":
            persona = "Data Scientist"

        rows.append(
            {
                "Focus": focus,
                "Drink": drink,
                "Schedule": schedule,
                "Approach": approach,
                "Energy": energy,
                "Persona": persona,
            }
        )

    df = pd.DataFrame(rows)
    feature_cols = ["Focus", "Drink", "Schedule", "Approach", "Energy"]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoded = encoder.fit_transform(df[feature_cols])
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(encoded, df["Persona"])
    return encoder, model


MODEL_ENCODER, MODEL = build_persona_model()

# Bilingual labels while keeping the model's stable English values.
OPTION_LABELS = {
    "Select a focus": "Select a focus / اختر تركيزك",
    "Solving puzzles & logic": "Solving puzzles & logic / حل الألغاز والمنطق",
    "Numbers & patterns": "Numbers & patterns / الأرقام والأنماط",
    "Building smart systems": "Building smart systems / بناء أنظمة ذكية",
    "Design & creativity": "Design & creativity / التصميم والإبداع",
    "Exploring & breaking things": "Exploring & breaking things / الاستكشاف وكسر القواعد",
    "Select a drink": "Select a drink / اختر مشروبك",
    "Coffee": "Coffee / قهوة",
    "Tea": "Tea / شاي",
    "Select a schedule": "Select a schedule / اختر روتينك",
    "Night owl": "Night owl / بتسهر بالليل",
    "Early bird": "Early bird / بتصحى بدري",
    "Select an approach": "Select an approach / اختر أسلوبك",
    "Deep focus alone": "Deep focus alone / تركيز عميق لوحدك",
    "Brainstorming with others": "Brainstorming with others / عصف ذهني مع فريق",
    "Trial & error experiments": "Trial & error experiments / تجربة وخطأ",
    "Careful planning": "Careful planning / تخطيط دقيق",
    "Select your energy": "Select your energy / اختر طاقتك",
    "Curious & analytical": "Curious & analytical / فضولي وتحليلي",
    "Rebellious & curious": "Rebellious & curious / متمرد وفضولي",
    "Visionary & ambitious": "Visionary & ambitious / صاحب رؤية وطموح",
    "Expressive & visual": "Expressive & visual / تعبيري وبصري",
}


# ---------- Recommendation text ----------
def persona_details(persona_name):
    detail_map = {
        "Data Scientist": (
            "You think in numbers, patterns, and stories hidden inside data. "
            "/ بتفكر بالأرقام والأنماط والقصص المختبئة جوه البيانات. عندك ميول قوية للتحليل واتخاذ القرارات المبنية على الداتا."
        ),
        "Hacker": (
            "You're driven by curiosity, logic puzzles, and finding the cracks others miss. "
            "/ بتتحرك بالفضول وحل الألغاز المنطقية، وبتلاقي الثغرات اللي محدش بيلاحظها. مجالك في الأمن السيبراني وكسر الأنظمة لفهمها."
        ),
        "AI Engineer": (
            "You love building systems that learn and scale on their own. "
            "/ بتحب تبني أنظمة تتعلم وتكبر لوحدها. مكانك الطبيعي في بناء نماذج الذكاء الاصطناعي والأنظمة الذكية."
        ),
        "Designer": (
            "You see the world visually and care deeply about how things feel to use. "
            "/ بتشوف الدنيا بعين بصرية وبتهتم إزاي الحاجات حاسس بيها المستخدم. مكانك في تصميم المنتجات وتجارب المستخدم."
        ),
    }
    return detail_map.get(persona_name, detail_map["Data Scientist"])


PERSONA_LABELS = {
    "Data Scientist": "Data Scientist / عالم بيانات",
    "Hacker": "Hacker / هاكر",
    "AI Engineer": "AI Engineer / مهندس ذكاء اصطناعي",
    "Designer": "Designer / مصمم",
}

PERSONA_ICONS = {
    "Data Scientist": "📊",
    "Hacker": "🕶️",
    "AI Engineer": "🤖",
    "Designer": "🎨",
}


# ---------- Prediction layer ----------
def predict_persona(focus, drink, schedule, approach, energy):
    """Predict a tech persona using a lightweight decision tree model."""
    placeholders = {
        "Select a focus",
        "Select a drink",
        "Select a schedule",
        "Select an approach",
        "Select your energy",
    }
    if any(value in placeholders for value in [focus, drink, schedule, approach, energy]):
        return "Please choose all five answers to generate your prediction. / من فضلك اختر الإجابات الخمسة للحصول على توقعك."

    input_df = pd.DataFrame(
        [
            {
                "Focus": focus,
                "Drink": drink,
                "Schedule": schedule,
                "Approach": approach,
                "Energy": energy,
            }
        ]
    )
    encoded = MODEL_ENCODER.transform(input_df[["Focus", "Drink", "Schedule", "Approach", "Energy"]])
    predicted_persona = MODEL.predict(encoded)[0]
    return predicted_persona


# ---------- Visual styling ----------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

        :root {
            --ink: #17212b;
            --muted: #66727c;
            --paper: #f7f8f5;
            --card: #ffffff;
            --navy: #1a1a2e;
            --teal: #00b8a9;
            --coral: #f26b5e;
            --gold: #f5b942;
            --line: #e3e8e4;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 12%, rgba(0, 184, 169, .16), transparent 23rem),
                radial-gradient(circle at 92% 8%, rgba(242, 107, 94, .10), transparent 24rem),
                var(--paper);
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }

        header[data-testid="stHeader"] {
            background: #ffffff !important;
            border-bottom: 1px solid var(--line);
        }

        header[data-testid="stHeader"] button,
        header[data-testid="stHeader"] svg,
        div[data-testid="stToolbar"] button,
        div[data-testid="stToolbar"] svg {
            color: var(--navy) !important;
            fill: var(--navy) !important;
        }

        div[data-testid="stToolbar"] {
            background: #ffffff !important;
        }

        .block-container {
            max-width: 1180px;
            padding: 2.5rem 2rem 4rem;
        }

        .brand {
            display: inline-flex;
            align-items: center;
            gap: .6rem;
            color: var(--navy);
            font-weight: 700;
            letter-spacing: .02em;
            font-size: 1.05rem;
        }

        .brand-mark {
            width: 34px;
            height: 34px;
            display: grid;
            place-items: center;
            background: var(--navy);
            color: #fff;
            border-radius: 11px 11px 11px 3px;
            font-size: 1.15rem;
            box-shadow: 0 8px 18px rgba(26, 26, 46, .2);
        }

        .eyebrow {
            color: var(--teal);
            text-transform: uppercase;
            font-size: .75rem;
            font-weight: 700;
            letter-spacing: .16em;
            margin: 3.5rem 0 1rem;
        }

        h1, h2, h3 {
            font-family: 'Playfair Display', Georgia, serif !important;
            color: var(--ink) !important;
        }

        .hero-title {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: clamp(3rem, 6vw, 5.5rem);
            line-height: .98;
            letter-spacing: -.045em;
            max-width: 750px;
            margin: 0;
            color: var(--ink);
        }

        .hero-title em {
            color: var(--coral);
            font-style: normal;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.7;
            max-width: 650px;
            margin: 1.4rem 0 0;
        }

        .hero-note {
            display: flex;
            align-items: center;
            gap: .65rem;
            margin-top: 1.6rem;
            color: var(--navy);
            font-size: .88rem;
            font-weight: 600;
        }

        .hero-note span {
            display: grid;
            place-items: center;
            width: 25px;
            height: 25px;
            border-radius: 50%;
            background: rgba(245, 185, 66, .22);
            color: #a66a00;
        }

        .section-label {
            color: var(--navy);
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 1.65rem;
            font-weight: 700;
            margin: 3rem 0 .45rem;
        }

        .section-help {
            color: var(--muted);
            margin: 0 0 1.25rem;
            font-size: .95rem;
        }

        div[data-testid='stForm'] {
            background: rgba(255, 255, 255, .88);
            border: 1px solid rgba(227, 232, 228, .95);
            border-radius: 24px;
            padding: 1.5rem 1.5rem .9rem;
            box-shadow: 0 20px 60px rgba(26, 26, 46, .08);
        }

        label[data-testid='stWidgetLabel'] p {
            color: var(--ink);
            font-weight: 700;
            font-size: .91rem;
        }

        div[data-baseweb='select'] > div {
            min-height: 48px;
            border-radius: 12px;
            border: 1px solid #9aaab4 !important;
            background: #fff !important;
            color: var(--ink) !important;
        }

        div[data-baseweb='select'] > div:hover {
            border-color: var(--teal);
        }

        div[data-baseweb='select'] input,
        div[data-baseweb='select'] [role='combobox'],
        div[data-baseweb='select'] [class*='singleValue'],
        div[data-baseweb='select'] [class*='placeholder'] {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            opacity: 1 !important;
        }

        div[data-baseweb='select'] svg {
            fill: var(--navy) !important;
            color: var(--navy) !important;
        }

        div[data-baseweb='popover'],
        div[data-baseweb='menu'] {
            background: #ffffff !important;
            border: 1px solid #c3cdd2 !important;
            color: var(--ink) !important;
            box-shadow: 0 12px 30px rgba(26, 26, 46, .18) !important;
        }

        div[data-baseweb='menu'] li,
        div[data-baseweb='menu'] [role='option'] {
            background: #ffffff !important;
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }

        div[data-baseweb='menu'] li:hover,
        div[data-baseweb='menu'] [role='option']:hover,
        div[data-baseweb='menu'] [aria-selected='true'] {
            background: #e6faf7 !important;
            color: var(--navy) !important;
            -webkit-text-fill-color: var(--navy) !important;
        }

        .stButton > button {
            width: 100%;
            min-height: 52px;
            margin-top: .8rem;
            border: 0;
            border-radius: 13px;
            background: var(--coral);
            color: #fff;
            font-size: 1rem;
            font-weight: 700;
            box-shadow: 0 10px 20px rgba(242, 107, 94, .22);
            transition: transform .16s ease-out, box-shadow .16s ease-out, background .16s ease-out;
        }

        .stButton > button:hover {
            background: #df594e;
            box-shadow: 0 14px 26px rgba(242, 107, 94, .3);
            transform: translateY(-2px);
        }

        .stButton > button:active {
            transform: scale(.97);
        }

        .result-card {
            display: flex;
            gap: 1rem;
            align-items: flex-start;
            margin-top: 1.4rem;
            padding: 1.35rem 1.5rem;
            border-radius: 18px;
            background: var(--navy);
            color: #fff;
            box-shadow: 0 16px 35px rgba(26, 26, 46, .22);
        }

        .result-icon {
            display: grid;
            place-items: center;
            flex: 0 0 44px;
            height: 44px;
            border-radius: 12px;
            background: var(--gold);
            color: var(--navy);
            font-size: 1.4rem;
        }

        .result-kicker {
            color: #9fe6dd;
            text-transform: uppercase;
            letter-spacing: .12em;
            font-size: .68rem;
            font-weight: 700;
            margin-bottom: .3rem;
        }

        .result-copy {
            color: #fff;
            font-size: 1rem;
            line-height: 1.55;
            margin: 0;
        }

        .side-card {
            background: var(--navy);
            color: #fff;
            border-radius: 24px;
            padding: 2rem;
            min-height: 100%;
            position: relative;
            overflow: hidden;
        }

        .side-card::after {
            content: '';
            position: absolute;
            width: 190px;
            height: 190px;
            border-radius: 50%;
            right: -75px;
            bottom: -75px;
            border: 28px solid rgba(0, 184, 169, .2);
        }

        .side-card h3 {
            color: #fff !important;
            font-size: 2rem;
            line-height: 1.05;
            margin: 0 0 1rem;
        }

        .side-card p {
            color: #c8d8df;
            line-height: 1.65;
            font-size: .92rem;
        }

        .side-rule {
            height: 1px;
            background: rgba(255,255,255,.16);
            margin: 1.8rem 0;
        }

        .side-stat {
            display: flex;
            justify-content: space-between;
            padding: .7rem 0;
            color: #dbe8ec;
            font-size: .87rem;
        }

        .side-stat strong {
            color: var(--gold);
            font-size: 1.1rem;
        }

        .footer {
            border-top: 1px solid var(--line);
            margin-top: 4rem;
            padding-top: 1.2rem;
            color: #849099;
            font-size: .78rem;
            display: flex;
            justify-content: space-between;
            gap: 1rem;
        }

        @media (max-width: 700px) {
            .block-container { padding: 1.5rem 1rem 3rem; }
            .eyebrow { margin-top: 2.5rem; }
            .hero-title { font-size: 3.35rem; }
            div[data-testid='stForm'] { padding: 1rem .85rem .5rem; }
            .side-card { margin-top: 1rem; }
            .footer { flex-direction: column; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Header / hero ----------
st.markdown(
    '<div class="brand"><div class="brand-mark">✦</div> PERSONALAB / معمل الشخصيات التقنية</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="eyebrow">Machine Learning demo / تجربة تعليم آلي</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="hero-title">Read your <em>tech mind.</em><br><span style="font-size:.45em; letter-spacing:0">اقرأ عقلك التقني.</span></h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-copy">Answer 5 quick questions and a Decision Tree model will guess the tech persona that fits you best.'
    '<br>جاوب على 5 أسئلة سريعة، وموديل Decision Tree هيتوقع شخصيتك التقنية اللي تناسبك أكتر.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-note"><span>✦</span> Takes less than a minute · Powered by scikit-learn / أقل من دقيقة · مدعوم بـ scikit-learn</div>',
    unsafe_allow_html=True,
)


# ---------- Inputs ----------
st.markdown('<div class="section-label">Tell us about yourself / احكيلنا عن نفسك</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-help">Choose the option that feels most like you right now. / اختر الخيار الأقرب إليك الآن.</p>',
    unsafe_allow_html=True,
)

with st.form("persona_form"):
    col1, col2 = st.columns(2, gap="large")
    with col1:
        focus = st.selectbox(
            "بتحب تحل المشاكل المعقدة ولا ترسم وتصمم؟ / What do you naturally focus on?",
            [
                "Select a focus",
                "Solving puzzles & logic",
                "Numbers & patterns",
                "Building smart systems",
                "Design & creativity",
                "Exploring & breaking things",
            ],
            format_func=lambda option: OPTION_LABELS[option],
        )
        schedule = st.selectbox(
            "بتسهر بالليل ولا بتصحى بدري؟ / What's your daily rhythm?",
            ["Select a schedule", "Night owl", "Early bird"],
            format_func=lambda option: OPTION_LABELS[option],
        )
        energy = st.selectbox(
            "إيه أقرب طاقة بتوصفك؟ / What's your general energy?",
            [
                "Select your energy",
                "Curious & analytical",
                "Rebellious & curious",
                "Visionary & ambitious",
                "Expressive & visual",
            ],
            format_func=lambda option: OPTION_LABELS[option],
        )
    with col2:
        drink = st.selectbox(
            "قهوة ولا شاي؟ / Coffee or tea?",
            ["Select a drink", "Coffee", "Tea"],
            format_func=lambda option: OPTION_LABELS[option],
        )
        approach = st.selectbox(
            "بتحب تشتغل إزاي؟ / How do you like to work?",
            [
                "Select an approach",
                "Deep focus alone",
                "Brainstorming with others",
                "Trial & error experiments",
                "Careful planning",
            ],
            format_func=lambda option: OPTION_LABELS[option],
        )

    submitted = st.form_submit_button("Predict my tech persona / توقّع شخصيتي التقنية  →")


if submitted:
    result = predict_persona(focus, drink, schedule, approach, energy)
    if result.startswith("Please choose all five answers"):
        st.warning(result)
    else:
        detail = persona_details(result)
        icon = PERSONA_ICONS.get(result, "✦")
        st.markdown(
            f'<div class="result-card"><div class="result-icon">{icon}</div>'
            f'<div><div class="result-kicker">Your tech persona / شخصيتك التقنية</div>'
            f'<p class="result-copy"><strong>{PERSONA_LABELS[result]}</strong><br>{detail}</p></div></div>',
            unsafe_allow_html=True,
        )


# ---------- Supporting panel ----------
st.markdown("<br>", unsafe_allow_html=True)
info_col, side_col = st.columns([1.45, .8], gap="large")
with info_col:
    st.markdown(
        '<div class="section-label" style="margin-top:0">How does this work? / إزاي ده شغال؟</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-copy" style="font-size:1rem; margin-top:.7rem;">'
        'Behind the scenes, a Decision Tree Classifier trained on sample data learns which combinations of '
        'answers tend to match each tech persona, then predicts yours in real time.'
        '<br>ورا الكواليس، فيه موديل Decision Tree Classifier اتدرب على بيانات نموذجية عشان يتعلم إيه توليفة الإجابات '
        'اللي بتوصل لكل شخصية تقنية، وبيتوقع شخصيتك أول ما تدوس على الزر.</p>',
        unsafe_allow_html=True,
    )
with side_col:
    st.markdown(
        '''<div class="side-card">
            <h3>Small answers.<br>Clear direction.<br><span style="font-size:.55em">إجابات صغيرة.<br>اتجاه واضح.</span></h3>
            <p>Your answers help the model personalize your result. / إجاباتك بتساعد الموديل يخصص نتيجتك.</p>
            <div class="side-rule"></div>
            <div class="side-stat"><span>Questions / الأسئلة</span><strong>05</strong></div>
            <div class="side-stat"><span>Time to complete / الوقت</span><strong>01:00</strong></div>
        </div>''',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer"><span>PersonaLab · Tech Persona Predictor / معمل الشخصيات · متنبئ الشخصية التقنية</span>'
    '<span>Made for curious minds / صُمم للعقول الفضولية</span></div>',
    unsafe_allow_html=True,
)