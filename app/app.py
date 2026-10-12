import logging
import sys
import base64
import html as html_lib
import json
from pathlib import Path

# Ajoute ai/src au chemin de recherche Python
sys.path.append(str(Path(__file__).parent.parent / "ai" / "src"))

# embeddings/engine ne sont PAS importés ici : ils chargent en cascade
# sentence-transformers, torch, pandas... (plusieurs secondes). Les importer
# au niveau module bloquerait l'affichage de la page de connexion, qui n'en
# a pourtant pas besoin. Voir load_model()/load_engine() ci-dessous.

import streamlit as st
from auth.database import (
    init_db,
    log_correction,
    create_session,
    get_session_user_id,
    rotate_session,
    get_user_by_id,
    add_chat_message,
    create_chat_conversation,
    delete_chat_conversation,
    get_chat_conversation,
    get_chat_messages,
    list_chat_conversations,
)
from auth.auth import register_user, login_user
from text_extraction import extract_text

_LOGGER = logging.getLogger(__name__)

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
        .st-key-sd_navigation[data-testid="stVerticalBlock"] {
            gap: 0 !important;
            row-gap: 0 !important;
        }
        .st-key-sd_navigation [data-testid="stElementContainer"] {
            margin: 0 !important;
            padding: 0 !important;
        }
        [class*="st-key-nav_page_"] button {
            width: 100%;
            justify-content: flex-start !important;
            background: #FFFFFF !important;
            border: 1px solid transparent !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            color: #374151 !important;
            font-size: 14.5px !important;
            font-weight: 500 !important;
            text-align: left !important;
            min-height: unset !important;
            padding: 9px 12px !important;
            margin: 0 !important;
        }
        [class*="st-key-nav_page_"] button > div {
            width: 100%;
            justify-content: flex-start !important;
            gap: 12px !important;
        }
        [class*="st-key-nav_page_"] button svg {
            color: #6B7280 !important;
        }
        .st-key-toggle_chat_history button {
            min-height: 2.5rem !important;
            padding: 6px !important;
            border: 1px solid transparent !important;
            border-radius: 8px !important;
            color: #6B7280 !important;
            box-shadow: none !important;
        }
        .st-key-toggle_chat_history button:hover {
            background: #F3F4F8 !important;
            color: var(--accent-dark) !important;
        }
        .st-key-sd_chat_history {
            padding: 2px 0 6px 12px;
            max-height: 260px;
            overflow-y: auto;
        }
        .st-key-sd_chat_history[data-testid="stVerticalBlock"] {
            gap: 4px !important;
            row-gap: 4px !important;
        }
        .st-key-sd_chat_history [data-testid="stElementContainer"] {
            margin: 0 !important;
            padding: 0 !important;
        }
        .st-key-new_chat_conversation button {
            min-height: 2.2rem !important;
            justify-content: flex-start !important;
            padding: 5px 10px !important;
            color: var(--accent-dark) !important;
            font-size: 13px !important;
            border: 1px solid transparent !important;
            box-shadow: none !important;
        }
        .st-key-new_chat_conversation button > div {
            width: 100%;
            justify-content: flex-start !important;
            gap: 8px !important;
        }
        .st-key-new_chat_conversation button:hover {
            background: #F3F4F8 !important;
        }
        .st-key-sd_chat_history [class*="st-key-chat_conversation_"] button {
            min-height: 2.2rem !important;
            justify-content: flex-start !important;
            padding: 5px 10px !important;
            color: #4B5563 !important;
            font-size: 13px !important;
            font-weight: 400 !important;
            border: 1px solid transparent !important;
            box-shadow: none !important;
        }
        .st-key-sd_chat_history [class*="st-key-chat_conversation_"] button > div {
            width: 100%;
            justify-content: flex-start !important;
        }
        .st-key-sd_chat_history [class*="st-key-chat_conversation_"] button:hover {
            background: #F3F4F8 !important;
        }
        .st-key-sd_chat_history [class*="st-key-chat_conversation_"] button[kind="primary"] {
            background: var(--accent-soft) !important;
            color: var(--accent-dark) !important;
        }
        .st-key-sd_chat_history [class*="st-key-delete_chat_"] [data-testid="stPopover"] button {
            width: 2rem !important;
            min-width: 2rem !important;
            min-height: 2rem !important;
            padding: 0 !important;
            border-radius: 50% !important;
            background: transparent !important;
            color: #9CA3AF !important;
            border: none !important;
            box-shadow: none !important;
            transition: background-color 120ms ease, color 120ms ease !important;
        }
        .st-key-sd_chat_history [class*="st-key-delete_chat_"] [data-testid="stPopover"] button svg {
            width: 17px !important;
            height: 17px !important;
        }
        .st-key-sd_chat_history [class*="st-key-delete_chat_"] [data-testid="stPopover"] button:hover {
            background: #FFF1F0 !important;
            color: #D45B55 !important;
        }
        [class*="st-key-nav_page_"] button:hover {
            background: #F3F4F8 !important;
            border-color: transparent !important;
            color: #374151 !important;
        }
        [class*="st-key-nav_page_"] button[kind="primary"] {
            background: var(--accent-soft) !important;
            color: var(--accent-dark) !important;
            font-weight: 600 !important;
        }
        [class*="st-key-nav_page_"] button[kind="primary"] svg {
            color: var(--accent-dark) !important;
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
        [class*="st-key-document_uploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
            display: none !important;
        }

        /* ---- Bouton icône (suppression d'un document) ---- */
        [class*="st-key-del_doc_"] button {
            padding: 4px 10px !important;
            border-radius: 8px !important;
            color: var(--muted) !important;
        }
        [class*="st-key-del_doc_"] button:hover {
            color: var(--danger) !important;
            border-color: var(--danger) !important;
        }

        /* ---- Menu ⋮ du dossier (renommer / supprimer) ----
           Le déclencheur du popover est un petit bouton icône seule, qui
           flotte sur l'en-tête du dossier : st.expander ne permet pas
           d'insérer un widget directement dans sa ligne d'en-tête
           (cliquable, gérée par Streamlit), donc on superpose la première
           ligne du contenu par un margin négatif. pointer-events: none sur
           la ligne évite de bloquer le clic sur l'en-tête ; réactivé juste
           sur le bouton. Le panneau du popover, lui, s'affiche normalement
           (positionné par Streamlit), aucun hack n'est nécessaire dessus. */
        [class*="st-key-cat_menu_"] [data-testid="stPopoverButton"] {
            padding: 2px 6px !important;
            min-height: 26px !important;
            border-radius: 8px !important;
            color: var(--muted) !important;
        }
        [class*="st-key-cat_menu_"] [data-testid="stPopoverButton"]:hover {
            color: var(--accent) !important;
            border-color: var(--accent) !important;
        }
        [class*="st-key-cat_menu_"] [data-testid="stPopoverButton"] svg {
            width: 16px !important;
            height: 16px !important;
        }
        [data-testid="stExpanderDetails"] [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:first-of-type [data-testid="stHorizontalBlock"] {
            margin-top: -63px;
            position: relative;
            z-index: 2;
            pointer-events: none;
            display: flex;
            align-items: center;
        }
        [data-testid="stExpanderDetails"] [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:first-of-type button {
            pointer-events: auto;
        }

        /* Dans le panneau du menu ⋮ : l'action "Supprimer" est en rouge. */
        [class*="st-key-confirm_del_cat_"] button {
            color: var(--danger) !important;
            border-color: var(--danger) !important;
        }
        [class*="st-key-confirm_del_cat_"] button:hover {
            background: var(--danger) !important;
            color: #FFFFFF !important;
        }

        /* ---- Nom de document cliquable (façon lien) ----
           st.button rend un <button>, st.link_button rend un <a> : on style
           les deux de la même façon pour que PDF/DOCX/TXT soient identiques
           visuellement malgré des widgets différents en interne. */
        [class*="st-key-open_"] button,
        [class*="st-key-open_"] a {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 2px 4px !important;
            color: var(--ink) !important;
            font-weight: 400 !important;
            justify-content: flex-start !important;
            width: auto !important;
            text-decoration: none !important;
        }
        [class*="st-key-open_"] button:hover,
        [class*="st-key-open_"] a:hover {
            color: var(--accent) !important;
            text-decoration: underline !important;
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

    cookie_token = st.context.cookies.get("smartdoc_session")
    legacy_query_token = st.query_params.get("session")
    restored_user_id = (
        get_session_user_id(cookie_token) if cookie_token else None
    )
    legacy_token_to_rotate = None
    if restored_user_id is None and legacy_query_token:
        restored_user_id = get_session_user_id(legacy_query_token)
        if restored_user_id is not None:
            legacy_token_to_rotate = legacy_query_token

    if restored_user_id is not None:
        user = get_user_by_id(restored_user_id)
        if user is not None:
            if legacy_token_to_rotate:
                session_token = rotate_session(legacy_token_to_rotate)
                if session_token:
                    st.session_state.pending_session_cookie = session_token
                else:
                    restored_user_id = None
            if restored_user_id is not None:
                st.session_state.user_id = restored_user_id
                st.session_state.email = user["email"]
    elif cookie_token:
        st.session_state.clear_session_cookie = True

    if legacy_query_token:
        st.query_params.pop("session", None)


def render_session_cookie_action(action, token=None):
    token_json = json.dumps(token)
    st.html(
        f"""
        <p id="session-cookie-status">Mise à jour sécurisée de la session…</p>
        <script>
        (async () => {{
            const status = document.getElementById("session-cookie-status");
            try {{
                const response = await fetch("/__session/{action}", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ token: {token_json} }}),
                    credentials: "same-origin"
                }});
                if (!response.ok) {{
                    throw new Error("Session request rejected");
                }}
                window.location.reload();
            }} catch (_) {{
                status.textContent = "Impossible de sécuriser la session. Recharge la page ou reconnecte-toi.";
            }}
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


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
                            st.session_state.pending_session_cookie = create_session(result)
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

PAGES = ["Import", "Mes documents", "Chatbot"]


def render_sidebar():

    user_id = st.session_state.user_id
    if st.session_state.get("chat_state_user_id") != user_id:
        st.session_state.chat_state_user_id = user_id
        st.session_state.active_chat_conversation_id = None
        st.session_state.chat_history_open = False
        st.query_params.pop("history", None)

    chat_history_open = st.query_params.get("history") == "open"
    st.session_state.chat_history_open = chat_history_open
    conversations = list_chat_conversations(user_id)
    conversation_ids = {conversation["id"] for conversation in conversations}
    requested_conversation = st.query_params.get("conversation")
    if requested_conversation and requested_conversation.isdigit():
        requested_id = int(requested_conversation)
        if requested_id in conversation_ids:
            st.session_state.active_chat_conversation_id = requested_id
    active_conversation_id = st.session_state.get("active_chat_conversation_id")
    if active_conversation_id not in conversation_ids:
        active_conversation_id = conversations[0]["id"] if conversations else None
        st.session_state.active_chat_conversation_id = active_conversation_id

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

        query_page = st.query_params.get("page")
        selected = st.session_state.get(
            "active_page",
            query_page if query_page in PAGES else PAGES[0]
        )
        page_icons = {
            "Import": ":material/upload:",
            "Mes documents": ":material/folder_open:",
            "Chatbot": ":material/chat:",
        }
        with st.container(key="sd_navigation"):
            for page in PAGES:
                page_key = page.lower().replace(" ", "_")
                if page == "Chatbot":
                    nav_col, add_col = st.columns([0.86, 0.14], gap="small")
                    with nav_col:
                        clicked = st.button(
                            page,
                            key=f"nav_page_{page_key}",
                            icon=page_icons[page],
                            type="primary" if page == selected else "secondary",
                            use_container_width=True,
                        )
                    with add_col:
                        show_conversations = st.button(
                            "",
                            key="toggle_chat_history",
                            icon=(
                                ":material/remove:"
                                if chat_history_open
                                else ":material/add:"
                            ),
                            help=(
                                "Masquer les conversations"
                                if chat_history_open
                                else "Afficher les conversations et en créer une nouvelle"
                            ),
                            use_container_width=True,
                        )
                    if clicked:
                        selected = page
                        st.session_state.active_page = page
                        st.query_params["page"] = page
                    if show_conversations:
                        st.session_state.active_page = page
                        selected = page
                        st.query_params["page"] = page
                        if chat_history_open:
                            st.query_params.pop("history", None)
                        else:
                            st.query_params["history"] = "open"
                        st.rerun()
                    if chat_history_open:
                        with st.container(key="sd_chat_history"):
                            if st.button(
                                "Nouvelle conversation",
                                key="new_chat_conversation",
                                icon=":material/add:",
                                use_container_width=True,
                            ):
                                active_conversation_id = create_chat_conversation(user_id)
                                conversations = list_chat_conversations(user_id)
                                st.session_state.active_chat_conversation_id = active_conversation_id
                                st.session_state.active_page = page
                                st.session_state.chat_history_open = True
                                selected = page
                                st.query_params["page"] = page
                                st.query_params["conversation"] = str(active_conversation_id)

                            if not conversations:
                                st.caption("Aucune conversation pour le moment.")
                            for conversation in conversations:
                                conversation_id = conversation["id"]
                                title = conversation["title"]
                                conversation_col, actions_col = st.columns(
                                    [0.86, 0.14],
                                    gap="small"
                                )
                                with conversation_col:
                                    if st.button(
                                        title,
                                        key=f"chat_conversation_{conversation_id}",
                                        icon=":material/chat_bubble_outline:",
                                        type=(
                                            "primary"
                                            if conversation_id == active_conversation_id
                                            else "secondary"
                                        ),
                                        use_container_width=True,
                                    ):
                                        active_conversation_id = conversation_id
                                        st.session_state.active_chat_conversation_id = conversation_id
                                        st.session_state.active_page = page
                                        selected = page
                                        st.query_params["page"] = page
                                        st.query_params["conversation"] = str(conversation_id)
                                with actions_col:
                                    with st.popover(
                                        "",
                                        key=f"delete_chat_{conversation_id}",
                                        icon=":material/delete_outline:",
                                        help="Supprimer cette discussion",
                                        use_container_width=True,
                                    ):
                                        st.caption("Cette discussion et ses messages seront supprimés.")
                                        if st.button(
                                            "Supprimer",
                                            key=f"confirm_delete_chat_{conversation_id}",
                                            type="primary",
                                            use_container_width=True,
                                        ):
                                            if delete_chat_conversation(user_id, conversation_id):
                                                remaining = list_chat_conversations(user_id)
                                                if conversation_id == active_conversation_id:
                                                    active_chat_id = (
                                                        remaining[0]["id"] if remaining else None
                                                    )
                                                    st.session_state.active_chat_conversation_id = active_chat_id
                                                    if active_chat_id is None:
                                                        st.query_params.pop("conversation", None)
                                                    else:
                                                        st.query_params["conversation"] = str(active_chat_id)
                                                st.rerun()
                                            else:
                                                st.error("Cette discussion n'existe plus.")
                else:
                    if st.button(
                        page,
                        key=f"nav_page_{page_key}",
                        icon=page_icons[page],
                        type="primary" if page == selected else "secondary",
                        use_container_width=True,
                    ):
                        selected = page
                        st.session_state.active_page = page
                        st.query_params["page"] = page

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
                st.session_state.clear_session_cookie = True
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
            label_visibility="collapsed",
            key="document_uploader"
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
                    raw_path.unlink(missing_ok=True)
                    return

                if not text.strip():
                    st.error("Aucun texte n'a pu être extrait de ce document.")
                    raw_path.unlink(missing_ok=True)
                    return

                # Le fichier original (pas une version .txt convertie) est
                # rangé tel quel : l'utilisateur doit pouvoir le retrouver
                # ensuite dans son format d'origine (PDF, DOCX...).
                with st.spinner("Analyse et classification en cours..."):
                    result = engine.categorize(str(raw_path))

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


DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def render_document_opener(category, doc):
    """
    Ouvre le document dans son format d'origine : un PDF s'ouvre dans un
    nouvel onglet (le navigateur l'affiche nativement), un DOCX se télécharge
    (le navigateur/l'OS l'ouvre ensuite avec Word) ; un .txt/.md garde
    l'aperçu texte, plus simple et suffisant pour ces formats.
    """

    filename = doc["filename"]
    extension = Path(filename).suffix.lower()
    key = f"open_{category}_{filename}"

    if extension == ".pdf":
        # st.markdown(unsafe_allow_html=True) supprime les attributs on* (pas
        # de JS exécutable) : impossible d'y mettre un onclick. Et Chrome/Edge
        # bloquent la navigation en plein onglet vers une URL data: (anti-
        # phishing), donc <a href="data:..." target="_blank"> ne marche pas
        # non plus. st.components.v1.html rend dans un vrai <iframe> qui,
        # lui, exécute le JS : on y construit un Blob et on ouvre son URL
        # blob:, qui n'est pas soumise à cette restriction.
        file_bytes = Path(doc["path"]).read_bytes()
        b64 = base64.b64encode(file_bytes).decode()
        safe_filename = html_lib.escape(filename)
        st.iframe(
            f"""
            <button id="btn" style="
                background: transparent; border: none; box-shadow: none;
                padding: 2px 4px; color: #1F2430; font-weight: 400;
                font-size: 14px; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                cursor: pointer; text-align: left;"
                onmouseover="this.style.color='#6C5CE7'; this.style.textDecoration='underline';"
                onmouseout="this.style.color='#1F2430'; this.style.textDecoration='none';"
            >📄 {safe_filename}</button>
            <script>
                document.getElementById('btn').addEventListener('click', function() {{
                    const byteChars = atob('{b64}');
                    const byteNumbers = new Array(byteChars.length);
                    for (let i = 0; i < byteChars.length; i++) {{ byteNumbers[i] = byteChars.charCodeAt(i); }}
                    const blob = new Blob([new Uint8Array(byteNumbers)], {{type: 'application/pdf'}});
                    window.open(URL.createObjectURL(blob), '_blank');
                }});
            </script>
            """,
            height=32,
        )

    elif extension == ".docx":
        file_bytes = Path(doc["path"]).read_bytes()
        st.download_button(
            f"📄 {filename}",
            data=file_bytes,
            file_name=filename,
            mime=DOCX_MIME,
            key=key,
        )

    else:
        if st.button(f"📄 {filename}", key=key):
            show_document_dialog(filename, doc["text"])


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

                _, col_menu = st.columns([7, 1])
                with col_menu:
                    with st.popover(
                        "",
                        icon=":material/more_vert:",
                        key=f"cat_menu_{category}",
                        help="Options du dossier",
                    ):
                        new_name = st.text_input(
                            "Renommer le dossier",
                            value=category,
                            key=f"rename_cat_{category}",
                        )
                        if st.button(
                            "Renommer",
                            icon=":material/edit:",
                            key=f"confirm_rename_cat_{category}",
                            use_container_width=True,
                        ):
                            result = engine.rename_category(category, new_name)
                            if "error" in result:
                                st.error(result["error"])
                            else:
                                st.toast(f"Renommé en « {result['to']} »", icon="✅")
                                st.rerun()

                        if st.button(
                            "Supprimer le dossier",
                            icon=":material/delete_outline:",
                            key=f"confirm_del_cat_{category}",
                            use_container_width=True,
                        ):
                            engine.delete_category(category)
                            st.rerun()

                docs = engine.df[engine.df["category"] == category]
                for _, doc in docs.iterrows():
                    filename = doc["filename"]
                    col_name, col_edit, col_delete = st.columns([7, 1, 1])

                    with col_name:
                        render_document_opener(category, doc)

                    with col_edit:
                        with st.popover("", icon=":material/edit:", help="Corriger la catégorie"):
                            st.markdown(f"**Corriger : {filename}**")

                            other_categories = [c for c in categories if c != category]
                            picked = st.selectbox(
                                "Catégorie existante",
                                options=["—"] + other_categories,
                                key=f"pick_cat_{category}_{filename}",
                            )
                            new_cat_free = st.text_input(
                                "ou nouvelle catégorie",
                                key=f"new_cat_{category}_{filename}",
                                placeholder="Nom de la nouvelle catégorie",
                            )

                            if st.button("Valider", key=f"confirm_correct_{category}_{filename}", type="primary"):
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
                        if st.button(
                            "",
                            icon=":material/delete_outline:",
                            key=f"del_doc_{category}_{filename}",
                            help="Supprimer ce document",
                        ):
                            engine.delete_document(filename, category)
                            st.rerun()


# ============================================================
# PAGE : CHATBOT
# ============================================================

def render_chatbot_page(engine):

    user_id = st.session_state.user_id
    conversation_id = st.session_state.get("active_chat_conversation_id")
    conversation = (
        get_chat_conversation(user_id, conversation_id)
        if conversation_id is not None else None
    )
    if conversation is None:
        conversation_id = create_chat_conversation(user_id)
        st.session_state.active_chat_conversation_id = conversation_id
        st.query_params["conversation"] = str(conversation_id)
        st.rerun()

    st.markdown('<div class="page-title">Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Pose une question sur tes documents, en langage naturel.</div>',
        unsafe_allow_html=True
    )

    messages = get_chat_messages(user_id, conversation_id)
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg["sources"]:
                st.caption(f"📚 Sources : {', '.join(msg['sources'])}")

    question = st.chat_input("Ta question...", key=f"chat_input_{conversation_id}")

    if question:
        title = None
        if not messages:
            try:
                title = engine.suggest_conversation_title(question)
            except Exception:
                _LOGGER.warning(
                    "Could not generate a title for chat conversation %s",
                    conversation_id,
                    exc_info=True,
                )

        if not add_chat_message(
            user_id,
            conversation_id,
            "user",
            question,
            title=title
        ):
            st.error("Cette conversation n'est plus disponible.")
            return

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans tes documents..."):
                response = engine.answer(
                    question,
                    conversation_history=messages[-8:]
                )

            st.write(response["answer"])
            if response["sources"]:
                st.caption(f"📚 Sources : {', '.join(response['sources'])}")

        if not add_chat_message(
            user_id,
            conversation_id,
            "assistant",
            response["answer"],
            response["sources"]
        ):
            st.error("La réponse a été générée, mais n'a pas pu être enregistrée dans cette conversation.")


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

if st.session_state.get("clear_session_cookie"):
    render_session_cookie_action("clear")
    st.stop()
elif st.session_state.get("pending_session_cookie"):
    render_session_cookie_action(
        "set",
        st.session_state.pending_session_cookie,
    )
    st.stop()
elif st.session_state.user_id is None:
    show_login_page()
else:
    show_main_app()
