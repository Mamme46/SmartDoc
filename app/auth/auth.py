import bcrypt
from auth.database import create_user, get_user_by_email


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def register_user(email: str, password: str):
    """
    Retourne (succès: bool, message: str)
    """

    existing = get_user_by_email(email)
    if existing:
        return False, "Un compte existe déjà avec cet email."

    if len(password) < 6:
        return False, "Le mot de passe doit contenir au moins 6 caractères."

    password_hash = hash_password(password)
    user_id = create_user(email, password_hash)

    return True, user_id


def login_user(email: str, password: str):
    """
    Retourne (succès: bool, user_id ou message d'erreur)
    """

    user = get_user_by_email(email)

    if not user:
        return False, "Email introuvable."

    if not verify_password(password, user["password_hash"]):
        return False, "Mot de passe incorrect."

    return True, user["id"]