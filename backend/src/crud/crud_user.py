from sqlalchemy.orm import Session
from src.models.user import User
from src.schemas.user_schema import UserCreate
from src.core.security import get_password_hash

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = get_password_hash(user.password)
    db_obj = User(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        is_superuser=user.is_superuser,
        is_active=user.is_active,
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj
