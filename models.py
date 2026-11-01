"""Database models for optional users and saved prediction scans."""

from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from extensions import db


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(320), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    language = db.Column(db.String(10), nullable=False, default="en")
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)

    scans = db.relationship(
        "Scan",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy=True,
    )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    top_disease = db.Column(db.String(200), nullable=False)
    top_confidence = db.Column(db.Float, nullable=False)
    prediction_set = db.Column(db.JSON, nullable=False, default=list)
    set_size = db.Column(db.Integer, nullable=False, default=0)
    thumbnail_path = db.Column(db.String(500), nullable=True)

    user = db.relationship("User", back_populates="scans")
