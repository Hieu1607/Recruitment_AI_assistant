import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum as SqlEnum, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.enums import RoleName, UserStatus


_ENUM_VALUES = lambda enum_cls: [item.value for item in enum_cls]


class UserAccount(Base):
    __tablename__ = "user_accounts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[UserStatus] = mapped_column(
        SqlEnum(UserStatus, name="user_status_enum", values_callable=_ENUM_VALUES),
        nullable=False,
        default=UserStatus.ACTIVE,
        server_default=UserStatus.ACTIVE.value,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    role_assignments: Mapped[list["RoleAssignment"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class RoleAssignment(Base):
    __tablename__ = "role_assignments"
    __table_args__ = (
        UniqueConstraint("user_id", "role_name", name="uq_role_assignment_user_role"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user_accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role_name: Mapped[RoleName] = mapped_column(
        SqlEnum(RoleName, name="role_name_enum", values_callable=_ENUM_VALUES),
        nullable=False,
    )
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    user: Mapped[UserAccount] = relationship(back_populates="role_assignments")
