import sys
from pathlib import Path

# Ajoute ai/src au chemin de recherche Python
sys.path.append(str(Path(__file__).parent.parent / "ai" / "src"))

# embeddings/engine ne sont PAS importés ici : ils chargent en cascade
# sentence-transformers, torch, pandas... (plusieurs secondes). Les importer
# au niveau module bloquerait l'affichage de la page de connexion, qui n'en
# a pourtant pas besoin. Voir load_model()/load_engine() ci-dessous.

import streamlit as st
from streamlit_option_menu import option_menu
from auth.database import init_db, log_correction
from auth.auth import register_user, login_user
from text_extraction import extract_text

init_db()

st.set_page_config(page_title="SmartDoc", page_icon="📁", layout="wide")


# ============================================================
# IDENTITÉ VISUELLE
# ============================================================

def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        :root {
            --accent: #6C5CE7;
            --accent-dark: #5A4BD1;
            --accent-soft: #EEEBFD;
            --ink: #1F2430;
            --muted: #6B7280;
            --border: #E5E7EF;
            --bg: #F7F8FC;
            --card: #FFFFFF;
            --danger: #E4573D;
        }

        /* ---- Cacher l'habillage Streamlit par défaut ----
           Le bouton pour rouvrir la sidebar repliée (stExpandSidebarButton)
           vit dans le même conteneur stToolbar que le menu/Deploy : on cible
           donc ces deux éléments précisément plutôt que tout le toolbar. */
        footer,
        [data-testid="stAppDeployButton"],
        [data-testid="stMainMenu"],
        [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
            visibility: hidden;
            height: 0;
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--bg);
        }

        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            max-width: 980px;
        }

        h1, h2, h3, h4 { color: var(--ink); font-weight: 700; }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            display: flex;
            flex-direction: column;
            height: 100%;
            padding-top: 1.25rem;
        }

        .sd-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0 4px 4px 4px;
        }
        .sd-brand .sd-logo {
            width: 30px; height: 30px;
            border-radius: 8px;
            background: var(--accent);
            display: flex; align-items: center; justify-content: center;
            font-size: 15px;
        }
        .sd-brand .sd-name {
            font-size: 16.5px; font-weight: 700; color: var(--ink); letter-spacing: -0.2px;
        }

        .sd-section-label {
            font-size: 12px;
            color: var(--muted);
            padding: 18px 6px 8px 6px;
        }

        .st-key-sd_footer {
            margin-top: auto;
            border-top: 1px solid var(--border);
            padding-top: 12px;
        }
        .sd-user-row {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 4px 4px 10px 4px;
        }
        .sd-avatar {
            width: 26px; height: 26px;
            border-radius: 50%;
            background: var(--accent-soft);
            color: var(--accent-dark);
            font-size: 12px; font-weight: 700;
            display: flex; align-items: center; justify-content: center;
            flex-shrink: 0;
        }
        .sd-user-email {
            font-size: 12.5px;
            color: var(--muted);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .st-key-sd_footer .stButton button {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--muted);
            font-size: 13px;
            font-weight: 500;
            border-radius: 8px;
            padding: 6px 10px;
        }
        .st-key-sd_footer .stButton button:hover {
            background: #F3F4F8;
            border-color: #D7D9E2;
            color: var(--ink);
        }

        /* ---- Cartes ---- */
        [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
            border-radius: 14px !important;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--card);
            box-shadow: 0 1px 2px rgba(20, 22, 31, 0.04);
            overflow: hidden;
            margin-bottom: 10px;
        }
        div[data-testid="stExpander"] summary {
            padding: 4px 2px;
            font-weight: 600;
            color: var(--ink);
        }
        div[data-testid="stExpander"] summary:hover { color: var(--accent); }

        .page-title {
            font-size: 26px;
            font-weight: 800;
            color: var(--ink);
            margin-bottom: 2px;
        }
        .page-subtitle {
            color: var(--muted);
            font-size: 14.5px;
            margin-bottom: 24px;
        }
        .section-label {
            font-size: 12.5px;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--muted);
            margin: 28px 0 10px 2px;
        }

        /* ---- Boutons primaires ---- */
        button[kind="primary"], button[kind="primaryFormSubmit"] {
            background: var(--accent) !important;
            border: none !important;
            border-radius: 9px !important;
            font-weight: 600;
        }
        button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
            background: var(--accent-dark) !important;
        }
        button[kind="secondary"], button[kind="secondaryFormSubmit"] {
            border-radius: 9px !important;
        }

        /* ---- Champs de saisie ----
           Filet de sécurité : le [theme] de .streamlit/config.toml force déjà
           un thème clair, mais si le navigateur a mémorisé une préférence
           sombre (choisie avant que le menu ne soit masqué), on la corrige ici. */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            background: var(--card) !important;
            color: var(--ink) !important;
            border: 1px solid var(--border) !important;
        }
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder {
            color: var(--muted) !important;
            opacity: 1;
        }
        /* Bouton "afficher le mot de passe" (icône oeil) : le composant interne
           de Streamlit force son propre fond sombre peu importe le thème, donc
           on écrase TOUS les éléments internes du champ (approche large mais
           strictement limitée à l'intérieur de stTextInput). */
        [data-testid="stTextInput"] * {
            background-color: var(--card) !important;
        }
        [data-testid="stTextInput"] input {
            color: var(--ink) !important;
            border: none !important;
        }
        [data-testid="stTextInput"] svg,
        [data-testid="stTextInput"] svg path {
            fill: var(--muted) !important;
        }
        /* Infobulle native "Press Enter to submit form" : Streamlit la
           positionne en absolu par-dessus le bas du champ au lieu d'en
           dessous (défaut natif, amplifié ici par le fond blanc qu'on vient
           de forcer sur tous les enfants du champ) → on la masque. */
        [data-testid="InputInstructions"] {
            display: none !important;
        }
        [data-testid="stTextInput"] > div {
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }

        /* ---- Onglets Connexion / Inscription ----
           L'onglet inactif hérite d'une couleur de texte blanche du thème
           sombre par défaut du navigateur : invisible sur fond clair. */
        [data-testid="stTabs"] button[role="tab"] {
            color: var(--danger) !important;
            opacity: 0.55;
        }
        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: var(--danger) !important;
            opacity: 1;
        }

        [data-testid="stFileUploaderDropzone"] {
            border-radius: 12px;
            background: #FBFAFF;
            border: 1.5px dashed #D6D2F5;
        }

        /* ---- Boutons icône (suppression) ---- */
        [class*="st-key-del_doc_"] button {
            padding: 4px 10px !important;
            border-radius: 8px !important;
        }

        /* ---- Nom de document cliquable (façon lien) ---- */
        [class*="st-key-open_"] button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 2px 4px !important;
            color: var(--ink) !important;
            font-weight: 400 !important;
            justify-content: flex-start !important;
            width: auto !important;
        }
        [class*="st-key-open_"] button:hover {
            color: var(--accent) !important;
            text-decoration: underline;
        }

        /* ---- Chatbot : style conversationnel façon ChatGPT/Claude ----
           Par défaut, stChatMessage affiche un avatar + un texte dont la
           couleur suit le thème sombre du navigateur (illisible sur fond
           clair). On masque les avatars, on aligne l'utilisateur à droite
           dans une bulle, et la réponse de l'assistant à gauche en texte
           simple, en forçant systématiquement une couleur de texte lisible. */
        [data-testid="stChatMessage"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 4px 0 !important;
        }
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"] {
            display: none !important;
        }
        [data-testid="stChatMessageContent"] {
            color: var(--ink) !important;
        }
        [data-testid="stChatMessageContent"] p,
        [data-testid="stChatMessageContent"] li,
        [data-testid="stChatMessageContent"] span,
        [data-testid="stChatMessageContent"] code {
            color: var(--ink) !important;
        }
        [data-testid="stChatMessageContent"] [data-testid="stCaptionContainer"],
        [data-testid="stChatMessageContent"] [data-testid="stCaptionContainer"] * {
            color: var(--muted) !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            justify-content: flex-end !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
            [data-testid="stChatMessageContent"] {
            background: var(--accent-soft) !important;
            border-radius: 18px !important;
            padding: 10px 16px !important;
            max-width: 72%;
            margin-left: auto;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
            [data-testid="stChatMessageContent"] {
            background: transparent !important;
            padding: 4px 2px !important;
            max-width: 100%;
        }

        /* Barre de saisie flottante en bas de page.
           Comme pour stTextInput, le composant interne garde un fond sombre
           venu du thème du navigateur si on ne force pas TOUS ses éléments
           internes (le conteneur, pas seulement le textarea) ; le bouton
           d'envoi (rond, accentué) est explicitement exclu pour garder sa
           couleur. */
        [data-testid="stBottomBlockContainer"] {
            background: var(--bg) !important;
        }
        [data-testid="stChatInput"] {
            background-color: var(--card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
        }
        [data-testid="stChatInput"] *:not(button):not(button *) {
            background-color: var(--card) !important;
        }
        [data-testid="stChatInputTextArea"] {
            color: var(--ink) !important;
        }
        [data-testid="stChatInputTextArea"]::placeholder {
            color: var(--muted) !important;
            opacity: 1;
        }
    </style>
    """, unsafe_allow_html=True)


inject_css()

if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.email = None


# ============================================================
# PAGE DE CONNEXION
# ============================================================

def show_login_page():

    _, center, _ = st.columns([1, 1.1, 1])

    with center:
        st.markdown(
            """
            <div style="text-align:center; margin: 48px 0 28px 0;">
                <div style="font-size: 40px;">📁</div>
                <div style="font-size: 26px; font-weight: 800; color: var(--ink); margin-top: 6px;">
                    Smart<span style="color: var(--accent);">Doc</span>
                </div>
                <div style="color: var(--muted); font-size: 14px; margin-top: 4px;">
                    Classement intelligent de tes documents
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.container(border=True):
            tab_login, tab_register = st.tabs(["Connexion", "Inscription"])

            with tab_login:
                with st.form("login_form"):
                    email = st.text_input("Email", placeholder="Adresse e-mail")
                    password = st.text_input("Mot de passe", type="password", placeholder="Mot de passe")
                    submitted = st.form_submit_button("Se connecter", type="primary", use_container_width=True)

                    if submitted:
                        success, result = login_user(email, password)
                        if success:
                            st.session_state.user_id = result
                            st.session_state.email = email
                            st.rerun()
                        else:
                            st.error(result)

            with tab_register:
                with st.form("register_form"):
                    email = st.text_input("Email", key="reg_email", placeholder="Adresse e-mail")
                    password = st.text_input("Mot de passe", type="password", key="reg_password", placeholder="6 caractères minimum")
                    password_confirm = st.text_input("Confirmer le mot de passe", type="password", placeholder="Confirmer le mot de passe")
                    submitted = st.form_submit_button("Créer un compte", type="primary", use_container_width=True)

                    if submitted:
                        if password != password_confirm:
                            st.error("Les mots de passe ne correspondent pas.")
                        else:
                            success, result = register_user(email, password)
                            if success:
                                st.success("Compte créé ! Tu peux maintenant te connecter.")
                            else:
                                st.error(result)


# ============================================================
# MOTEUR (cache)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """Chargé une seule fois pour toute l'app, peu importe l'utilisateur."""
    from embeddings import get_embedding_model
    return get_embedding_model()


@st.cache_resource(show_spinner="Chargement de ton espace...")
def load_engine(user_id):
    """
    Un moteur par utilisateur. Mis en cache par user_id : Streamlit ne
    recrée l'engine (et ne recalcule les embeddings) que si le user_id change.
    """
    from engine import SmartDocEngine
    model = load_model()
    user_data_path = Path(__file__).parent / "data" / "users" / str(user_id)
    return SmartDocEngine(data_path=user_data_path, embedding_model=model)


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:
        st.markdown(
            """
            <div class="sd-brand">
                <div class="sd-logo">📁</div>
                <div class="sd-name">SmartDoc</div>
            </div>
            <div class="sd-section-label">Espace de travail</div>
            """,
            unsafe_allow_html=True
        )

        selected = option_menu(
            menu_title=None,
            options=["Import", "Mes documents", "Chatbot"],
            icons=["upload", "folder2-open", "chat-dots"],
            default_index=0,
            styles={
                "container": {"padding": "0", "background-color": "#FFFFFF"},
                "icon": {"color": "#6B7280", "font-size": "15px"},
                "nav-link": {
                    "font-size": "14.5px",
                    "font-weight": "500",
                    "text-align": "left",
                    "margin": "2px 0",
                    "padding": "9px 12px",
                    "border-radius": "8px",
                    "color": "#374151",
                    "background-color": "#FFFFFF",
                    "--hover-color": "#F3F4F8",
                },
                "nav-link-selected": {
                    "background-color": "#EEEBFD",
                    "color": "#5A4BD1",
                    "font-weight": "600",
                },
            },
        )

        with st.container(key="sd_footer"):
            initial = (st.session_state.email or "?")[0].upper()
            st.markdown(
                f"""
                <div class="sd-user-row" title="{st.session_state.email}">
                    <div class="sd-avatar">{initial}</div>
                    <div class="sd-user-email">{st.session_state.email}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("↩  Déconnexion", key="logout_btn", use_container_width=True):
                st.session_state.user_id = None
                st.session_state.email = None
                st.rerun()

    return selected


# ============================================================
# PAGE : IMPORT
# ============================================================

def render_import_page(engine):

    st.markdown('<div class="page-title">Import</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Importe un document : il sera classé automatiquement par similarité de contenu.</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        st.markdown("**📤 Importer un nouveau document**")
        uploaded_file = st.file_uploader(
            "Dépose un document (.txt, .md, .pdf, .docx)",
            type=["txt", "md", "pdf", "docx"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:

            if st.button("Classer ce document", type="primary"):

                temp_dir = Path(__file__).parent / "data" / "temp_uploads"
                temp_dir.mkdir(parents=True, exist_ok=True)
                raw_path = temp_dir / uploaded_file.name

                with open(raw_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    with st.spinner("Extraction du texte..."):
                        text = extract_text(raw_path)
                except Exception as exc:
                    st.error(f"Impossible de lire ce fichier : {exc}")
                    return
                finally:
                    raw_path.unlink(missing_ok=True)

                if not text.strip():
                    st.error("Aucun texte n'a pu être extrait de ce document.")
                    return

                temp_path = raw_path.with_suffix(".txt")
                temp_path.write_text(text, encoding="utf-8")

                with st.spinner("Analyse et classification en cours..."):
                    result = engine.categorize(str(temp_path))

                if "error" in result:
                    st.error(result["error"])
                else:
                    if result["action"] == "new_category":
                        st.success(
                            f"🆕 Nouvelle catégorie créée : **{result['category']}** "
                            f"(score: {result['score']:.3f})"
                        )
                    else:
                        st.success(
                            f"📁 Rangé dans : **{result['category']}** "
                            f"(score: {result['score']:.3f})"
                        )


# ============================================================
# PAGE : MES DOCUMENTS (explorateur)
# ============================================================

@st.dialog("Aperçu du document", width="large")
def show_document_dialog(filename, text):
    st.markdown(f"**📄 {filename}**")
    st.text_area("Contenu", text, height=420, disabled=True, label_visibility="collapsed")


def render_documents_page(engine):

    st.markdown('<div class="page-title">Mes documents</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Clique sur un document pour l\'ouvrir.</div>',
        unsafe_allow_html=True
    )

    if len(engine.df) == 0:
        st.info("Aucun document importé pour l'instant.")
    else:
        categories = sorted(engine.df["category"].unique())

        for category in categories:
            doc_count = len(engine.df[engine.df["category"] == category])
            with st.expander(f"📁 {category}  ·  {doc_count} document{'s' if doc_count > 1 else ''}"):

                col_del, _ = st.columns([1, 3])
                with col_del:
                    if st.button("🗑️ Supprimer ce dossier", key=f"del_cat_{category}"):
                        engine.delete_category(category)
                        st.rerun()

                docs = engine.df[engine.df["category"] == category]
                for _, doc in docs.iterrows():
                    filename = doc["filename"]
                    col_name, col_edit, col_delete = st.columns([7, 1, 1])

                    with col_name:
                        if st.button(f"📄 {filename}", key=f"open_{filename}"):
                            show_document_dialog(filename, doc["text"])

                    with col_edit:
                        with st.popover("", icon=":material/edit:", help="Corriger la catégorie"):
                            st.markdown(f"**Corriger : {filename}**")

                            other_categories = [c for c in categories if c != category]
                            picked = st.selectbox(
                                "Catégorie existante",
                                options=["—"] + other_categories,
                                key=f"pick_cat_{filename}",
                            )
                            new_cat_free = st.text_input(
                                "ou nouvelle catégorie",
                                key=f"new_cat_{filename}",
                                placeholder="Nom de la nouvelle catégorie",
                            )

                            if st.button("Valider", key=f"confirm_correct_{filename}", type="primary"):
                                target = new_cat_free.strip() or (picked if picked != "—" else "")

                                if not target:
                                    st.warning("Choisis ou saisis une catégorie.")
                                else:
                                    result = engine.correct_classification(filename, category, target)
                                    if "error" in result:
                                        st.error(result["error"])
                                    else:
                                        log_correction(
                                            st.session_state.user_id, filename, category, target
                                        )
                                        st.toast(f"Déplacé vers {target}", icon="✅")
                                        st.rerun()

                    with col_delete:
                        if st.button("🗑️", key=f"del_doc_{filename}"):
                            engine.delete_document(filename)
                            st.rerun()


# ============================================================
# PAGE : CHATBOT
# ============================================================

def render_chatbot_page(engine):

    st.markdown('<div class="page-title">Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Pose une question sur tes documents, en langage naturel.</div>',
        unsafe_allow_html=True
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                st.caption(f"📚 Sources : {', '.join(msg['sources'])}")

    question = st.chat_input("Ta question...")

    if question:

        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans tes documents..."):
                response = engine.answer(question)

            st.write(response["answer"])
            if response["sources"]:
                st.caption(f"📚 Sources : {', '.join(response['sources'])}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"]
        })


# ============================================================
# APP PRINCIPALE
# ============================================================

def show_main_app():

    selected = render_sidebar()
    engine = load_engine(st.session_state.user_id)

    if selected == "Import":
        render_import_page(engine)
    elif selected == "Mes documents":
        render_documents_page(engine)
    else:
        render_chatbot_page(engine)


# ============================================================
# ROUTAGE PRINCIPAL
# ============================================================

if st.session_state.user_id is None:
    show_login_page()
else:
    show_main_app()
