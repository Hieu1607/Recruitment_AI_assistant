import os
import sys
from pathlib import Path
from typing import List

from sqlalchemy import select

# Allow running this script from backend/ with: python seeds/seed_initial_user_and_roles.py
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.models.enums import RoleName, UserStatus
from src.models.session import SessionLocal
from src.models.user_account import RoleAssignment, UserAccount


def _parse_roles(raw_roles: str) -> List[str]:
    values = [part.strip().lower() for part in raw_roles.split(",") if part.strip()]
    if not values:
        values = [RoleName.ADMIN.value]

    unique_values: List[str] = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)

    parsed_roles: List[str] = []
    valid_values = {role.value for role in RoleName}
    for value in unique_values:
        if value not in valid_values:
            raise ValueError(
                f"Invalid role '{value}'. Valid roles: {', '.join(sorted(valid_values))}"
            )
        parsed_roles.append(value)

    return parsed_roles


def seed_initial_user_and_roles() -> None:
    seed_email = os.getenv("SEED_USER_EMAIL", "admin@recruitment.local").strip().lower()
    seed_display_name = os.getenv("SEED_USER_DISPLAY_NAME", "Initial Admin").strip()
    seed_roles = _parse_roles(
        os.getenv("SEED_USER_ROLES", "admin,recruiter,viewer")
    )

    db = SessionLocal()
    try:
        user = db.execute(
            select(UserAccount).where(UserAccount.email == seed_email)
        ).scalar_one_or_none()

        created_user = False
        if user is None:
            user = UserAccount(
                email=seed_email,
                display_name=seed_display_name,
                status=UserStatus.ACTIVE.value,
            )
            db.add(user)
            db.flush()
            created_user = True

        added_roles: List[str] = []
        for role in seed_roles:
            existing_assignment = db.execute(
                select(RoleAssignment).where(
                    RoleAssignment.user_id == user.id,
                    RoleAssignment.role_name == role,
                )
            ).scalar_one_or_none()

            if existing_assignment is None:
                db.add(
                    RoleAssignment(
                        user_id=user.id,
                        role_name=role,
                    )
                )
                added_roles.append(role)

        db.commit()

        print("Seed completed successfully.")
        print(f"User: {user.email} ({user.display_name})")
        print(f"Created user: {created_user}")
        print(f"Roles requested: {', '.join(seed_roles)}")
        print(f"Roles added in this run: {', '.join(added_roles) if added_roles else 'none'}")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_initial_user_and_roles()
