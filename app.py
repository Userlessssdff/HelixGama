import os
import uuid
import zipfile
import io
import re
import datetime
import smtplib
import random
import string
import stripe
import time
import google.generativeai as genai
from functools import wraps
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, request, jsonify, send_file, render_template_string, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import json  # === AETHER FEATURE ADDITION: JSON Import ===

# --- CONFIGURATION ---
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# FIX: Use DATABASE_URL if it exists (Render sets this automatically), otherwise use SQLALCHEMY_DATABASE_URI
database_url = os.getenv('DATABASE_URL') or os.getenv('SQLALCHEMY_DATABASE_URI')
# Fix: Render uses 'postgres://' but SQLAlchemy 1.4+ requires 'postgresql://'
if database_url and database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

# Ensure API key is set before configuring the client
# Set stripe.api_key only if it exists, otherwise it will be skipped by checkout mock
stripe_secret_key = os.getenv('STRIPE_SECRET_KEY')
if stripe_secret_key:
    stripe.api_key = stripe_secret_key

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not set in .env. AI generation will fail.")

# Define the model to use globally
GEMINI_MODEL_NAME = 'gemini-2.5-flash-lite'  # Default
GEMINI_MODEL_PRO = 'gemini-2.5-flash-lite'  # Mapping 2.5 Pro requirement to working model or hypothetical endpoint

# TIER DEFINITIONS FOR UPGRADE LOGIC (ORDERED)
TIER_LEVELS = {
    'free': 0,
    'starter': 1,
    'premium': 2,  # Legacy support
    'developer': 3,
    'ultra': 4
}

# BAD WORDS LIST (MANDATORY FILTER)
BAD_WORDS = ['dick', 'fuck', 'curse', 'niggers', 'damn', 'scam', 'hate', 'nga', 'nigger', 'stupid', 'idiot', 'ugly',
             'offensive']

# --- GLOBAL LOCKS FOR API SAFETY ---
# Stores user_ids currently processing a generation request
GENERATION_LOCKS = set()
# Stores timestamp of last completion per user_id
USER_COOLDOWNS = {}


def check_content_safety(text):
    if not text: return True
    lower_text = text.lower()
    for w in BAD_WORDS:
        if w in lower_text: return False
    return True


# --- MODELS ---

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    # Tier: 'free', 'starter', 'premium', 'developer', 'ultra'
    tier = db.Column(db.String(20), default='free')
    games_generated = db.Column(db.Integer, default=0)  # Lifetime count (legacy/free)

    # Usage Tracking (Feature 4)
    last_usage_reset = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    daily_usage_normal = db.Column(db.Integer, default=0)
    daily_usage_pro = db.Column(db.Integer, default=0)

    # Security & Verification
    verification_code = db.Column(db.String(6))
    code_expiry = db.Column(db.DateTime)
    is_verified = db.Column(db.Boolean, default=False)

    failed_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime)

    games = db.relationship('Game', backref='author', lazy=True)


# New model to store billing info safely without modifying User schema dangerously
class UserBilling(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    plan_interval = db.Column(db.String(20))  # 'month' or 'year'
    subscription_end = db.Column(db.DateTime)


class Game(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    public_id = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()))
    title = db.Column(db.String(100))
    prompt = db.Column(db.Text)
    code_html = db.Column(db.Text)
    is_public = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    # === AETHER FEATURE ADDITION START ===
    # Feature 1: Store settings safely as nullable JSON
    # Stores: visual_overrides, global_settings (game_width), author_preference
    settings_json = db.Column(db.Text, nullable=True)
    # Feature 3: Play Count and Likes
    play_count = db.Column(db.Integer, default=0)
    like_count = db.Column(db.Integer, default=0)
    # === AETHER FEATURE ADDITION END ===


# New model for unique likes
class GameLike(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    game_id = db.Column(db.String(36), db.ForeignKey('game.id'), nullable=False)


# === AETHER FEATURE ADDITION: TEAM ACCESS ===
class GameCollaborator(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.String(36), db.ForeignKey('game.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    # New Columns for Invite System
    status = db.Column(db.String(20), default='pending')  # pending, accepted
    role = db.Column(db.String(20), default='admin')  # admin (edit only)


# CREATE TABLES AFTER ALL MODELS ARE DEFINED
with app.app_context():
    db.create_all()
    print("Database tables created successfully!")


# --- UTILS & EMAIL ---

def send_email(to_email, subject, body, code=None):
    host = os.getenv('EMAIL_SMTP_HOST')
    user = os.getenv('EMAIL_SMTP_USER')
    password = os.getenv('EMAIL_SMTP_PASS')

    # ALWAYS print the code to the terminal for easy testing
    if code:
        print(f"\n{'='*60}")
        print(f"[VERIFICATION CODE] To: {to_email}")
        print(f"[VERIFICATION CODE] Subject: {subject}")
        print(f"[VERIFICATION CODE] CODE: {code}")
        print(f"{'='*60}\n")
    else:
        print(f"\n[EMAIL] To: {to_email} | Subject: {subject}\n")

    if not host or not user or not password:
        print(f"WARNING: SMTP not fully configured. Missing: host={bool(host)}, user={bool(user)}, pass={bool(password)}")
        return True

    try:
        print(f"Attempting SMTP connection to {host}:{os.getenv('EMAIL_SMTP_PORT', 587)}...")
        msg = MIMEMultipart()
        msg['From'] = user
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(host, int(os.getenv('EMAIL_SMTP_PORT', 587)))
        server.set_debuglevel(1)  # Enable debug output
        server.starttls()
        print(f"Logging in as {user}...")
        server.login(user, password)
        print("Login successful! Sending message...")
        server.send_message(msg)
        server.quit()
        print("Email sent successfully!")
        return True
    except Exception as e:
        print(f"SMTP Email Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_code():
    return ''.join(random.choices(string.digits, k=6))


def clean_ai_response(text):
    """Extracts only the HTML code block from the AI's response."""
    # Pattern to find anything between ```...```, optionally preceded by a language hint (html)
    # This is a fallback if the system prompt fails to enforce plain text.
    pattern = r"```(?:html|javascript|css)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # If no markdown block is found, assume the entire response is the code
    return text.strip()


def generate_game_logic(prompt, model_name=None):
    if not GEMINI_API_KEY:
        raise Exception("API_KEY_MISSING")

    # Determine model to use
    selected_model = model_name if model_name else GEMINI_MODEL_NAME

    sys_prompt = (
        "You are AETHER, a deterministic HTML5 GAME ENGINE.\n"
        "You generate FULL, REAL, PLAYABLE GAMES — NOT demos, NOT mockups.\n\n"

        "ABSOLUTE OUTPUT RULES (MANDATORY, NO EXCEPTIONS):\n"
        "- The FIRST character of the response MUST be '<'\n"
        "- The response MUST start EXACTLY with <!DOCTYPE html>\n"
        "- The response MUST contain ONLY raw HTML code\n"
        "- DO NOT include explanations, titles, headers, labels, or comments outside HTML\n"
        "- DO NOT include markdown, code fences, or formatting\n"
        "- DO NOT include any text before or after the HTML document\n"
        "- Any violation makes the output INVALID\n\n"

        "MANDATORY GAME REQUIREMENTS:\n"
        "- Use HTML5 <canvas>\n"
        "- Implement a continuous requestAnimationFrame loop\n"
        "- Player-controlled entity (keyboard: WASD or Arrow Keys)\n"
        "- Enemies or obstacles with movement and interaction\n"
        "- Collision detection with visible consequences\n"
        "- Clear win AND lose conditions\n"
        "- Score, upgrades, or progression system\n"
        "- Game-over screen with restart functionality\n\n"

        "TECHNICAL REQUIREMENTS:\n"
        "- Single self-contained HTML file\n"
        "- Embedded CSS and JavaScript\n"
        "- No external assets, libraries, fonts, or CDNs\n"
        "- Must work fully offline\n"
        "- Optimized for iframe embedding\n"
        "- Stable performance (no infinite memory growth)\n\n"

        "DESIGN REQUIREMENTS:\n"
        "- Smooth gameplay (delta-time based movement)\n"
        "- Responsive canvas scaling\n"
        "- Mobile and desktop compatible controls\n"
        "- Clean visual style (no placeholder UI)\n\n"

        "FAILURE POLICY:\n"
        "- If the user prompt is vague, YOU must design a complete game anyway\n"
        "- If unsure, choose sensible defaults and continue\n"
        "- NEVER ask questions\n\n"

        "REMINDER:\n"
        "This is RAW CODE GENERATION ONLY. Output the game code and nothing else."
    )

    model = genai.GenerativeModel(
        model_name=selected_model,
        system_instruction=sys_prompt
    )

    try:
        print(f"[{datetime.datetime.utcnow()}] Sending SINGLE prompt to Gemini ({selected_model})...")

        # Explicitly disable retries in generation_config if possible or rely on default single call behavior
        # We assume standard generate_content is synchronous and single-shot
        res = model.generate_content(prompt)

        if not res.text:
            raise Exception("AI_RESPONSE_EMPTY")

        cleaned_code = clean_ai_response(res.text)

        if not cleaned_code.startswith("<!DOCTYPE html>"):
            print(f"Warning: Cleaned code doesn't start with <!DOCTYPE html>. Output was:\n{cleaned_code[:500]}...")

        return cleaned_code

    except Exception as e:
        print(f"--- AI Generation Fatal Error ---: {e}")
        raise e


# --- USAGE LIMITS HELPER (Feature 4) ---
def check_and_reset_usage(user):
    now = datetime.datetime.utcnow()
    # Reset if more than 24 hours have passed
    if user.last_usage_reset and (now - user.last_usage_reset).total_seconds() > 86400:
        user.daily_usage_normal = 0
        user.daily_usage_pro = 0
        user.last_usage_reset = now
        db.session.commit()


def get_remaining_uses(user, model_type):
    check_and_reset_usage(user)

    # Limits Definition
    limits = {
        'free': {'normal': 1, 'pro': 0},
        # Legacy free limit logic handled separately usually, but mapped here for safety
        'starter': {'normal': 20, 'pro': 0},
        'premium': {'normal': 9999, 'pro': 0},  # Legacy Premium
        'developer': {'normal': 50, 'pro': 5},
        'ultra': {'normal': 1000, 'pro': 100}
    }

    tier_limits = limits.get(user.tier, limits['free'])

    if model_type == 'pro':
        return max(0, tier_limits['pro'] - user.daily_usage_pro)
    else:
        return max(0, tier_limits['normal'] - user.daily_usage_normal)


# --- ADMIN OVERRIDE HELPER ---
def apply_admin_overrides(user):
    """
    Applies admin override from upgrades_allowlist.json.
    If user is not in the allowlist, their tier resets to "free".
    """
    try:
        allowlist = {}
        if os.path.exists('upgrades_allowlist.json'):
            with open('upgrades_allowlist.json', 'r') as f:
                allowlist = json.load(f)

        if user.email in allowlist:
            override_tier = allowlist[user.email]
            if user.tier != override_tier:
                user.tier = override_tier
                db.session.commit()
                print(f"Admin Override Applied: {user.email} -> {override_tier}")
        else:
            if user.tier != "free":
                user.tier = "free"
                db.session.commit()
                print(f"Admin Override Removed: {user.email} -> free")
    except Exception as e:
        print(f"Override Error: {e}")


@login_manager.user_loader
def load_user(user_id):
    u = User.query.get(int(user_id))
    if u:
        apply_admin_overrides(u)
    return u


# --- FLASK-LOGIN USER LOADER ---
@login_manager.user_loader
def load_user(user_id):
    try:
        u = User.query.get(int(user_id))
        if u:
            apply_admin_overrides(u)
        return u
    except Exception as e:
        print(f"Load User Error: {e}")
        return None

# --- ROUTES: AUTHENTICATION ---

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()

    if not email:
        return jsonify({'success': False, 'message': 'Email Identity required.'}), 400
    if not password:
        return jsonify({'success': False, 'message': 'Access Key (Password) required.'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email Identity already exists.'}), 409

    code = generate_code()
    hashed_pw = generate_password_hash(password)
    new_user = User(
        email=email,
        password=hashed_pw,
        verification_code=code,
        code_expiry=datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
    )
    db.session.add(new_user)
    db.session.commit()

    send_email(email, "Verify your AETHER Account", f"Your verification code is: {code}", code)
    return jsonify({'success': True, 'require_verify': True, 'email': email})


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()

    if not email:
        return jsonify({'success': False, 'message': 'Email Identity required.'}), 400
    if not password:
        return jsonify({'success': False, 'message': 'Access Key (Password) required.'}), 400

    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

    if user.locked_until and user.locked_until > datetime.datetime.utcnow():
        wait_time = (user.locked_until - datetime.datetime.utcnow()).seconds
        return jsonify({'success': False, 'message': f'Account locked. Try again in {wait_time}s'}), 429

    if check_password_hash(user.password, password):
        user.failed_attempts = 0
        user.locked_until = None

        code = generate_code()
        user.verification_code = code
        user.code_expiry = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)

        # Apply Admin Overrides on Login
        apply_admin_overrides(user)

        db.session.commit()

        send_email(user.email, "AETHER Login Code", f"Your login code is: {code}", code)
        return jsonify({'success': True, 'require_verify': True, 'email': user.email})
    else:
        user.failed_attempts += 1
        lock_seconds = 0

        if user.failed_attempts >= 3:
            lock_seconds = 10 * user.failed_attempts
            user.locked_until = datetime.datetime.utcnow() + datetime.timedelta(seconds=lock_seconds)

        db.session.commit()

        msg = "Wrong password."
        if lock_seconds > 0:
            msg += f" Account locked for {lock_seconds} seconds."

        return jsonify({'success': False, 'message': msg}), 401


@app.route('/api/auth/verify', methods=['POST'])
def verify_code():
    data = request.json
    email = data.get('email')
    code = data.get('code')
    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404

    if user.verification_code == code and user.code_expiry and user.code_expiry > datetime.datetime.utcnow():
        user.is_verified = True
        user.verification_code = None
        db.session.commit()
        login_user(user)
        return jsonify({'success': True})

    return jsonify({'success': False, 'message': 'Invalid or expired code'}), 401


@app.route('/api/auth/forgot', methods=['POST'])
def forgot_pass():
    data = request.json
    email = data.get('email', '').strip()

    if not email:
        return jsonify({'success': False, 'message': 'Email Identity required for recovery.'}), 400

    user = User.query.filter_by(email=email).first()

    if user:
        code = generate_code()
        user.verification_code = code
        user.code_expiry = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
        db.session.commit()
        send_email(user.email, "Reset Password", f"Your reset code is: {code}", code)

    return jsonify({'success': True, 'require_verify': True, 'email': email,
                    'message': 'If account exists, verification code was sent.'})


@app.route('/api/auth/reset', methods=['POST'])
def reset_pass():
    data = request.json
    email = data.get('email')
    code = data.get('code')
    new_password = data.get('new_password')

    if not new_password or len(new_password) < 6:
        return jsonify({'success': False, 'message': 'New password must be at least 6 characters.'}), 400

    user = User.query.filter_by(email=email).first()

    if user and user.verification_code == code and user.code_expiry and user.code_expiry > datetime.datetime.utcnow():
        user.password = generate_password_hash(new_password)
        user.verification_code = None
        db.session.commit()
        return jsonify({'success': True})

    return jsonify({'success': False, 'message': 'Invalid or expired code for reset.'}), 401


@app.route('/api/auth/status')
def auth_status():
    if current_user.is_authenticated:
        # Check usage for UI
        check_and_reset_usage(current_user)
        return jsonify({
            'authenticated': True,
            'email': current_user.email,
            'tier': current_user.tier,
            'builds': current_user.games_generated,
            'usage': {
                'normal': current_user.daily_usage_normal,
                'pro': current_user.daily_usage_pro
            }
        })
    return jsonify({'authenticated': False})


@app.route('/api/auth/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'success': True})


# --- ROUTES: GAME & PAYMENT ---

@app.route('/')
def index():
    return send_file('index.html')


@app.route('/s/<public_id>')
def shared_game(public_id):
    game = Game.query.filter_by(public_id=public_id, is_public=True).first_or_404()
    # === AETHER FEATURE ADDITION START ===
    # Feature 3: Play Count on shared link open
    try:
        game.play_count += 1
        db.session.commit()
    except:
        pass  # Non-critical
    # === AETHER FEATURE ADDITION END ===
    return game.code_html


@app.route('/api/projects')
@login_required
def list_projects():
    # Own games
    games = Game.query.filter_by(user_id=current_user.id).order_by(Game.created_at.desc()).all()

    # Collaborating games (Accepted Only)
    collab_entries = GameCollaborator.query.filter_by(user_id=current_user.id, status='accepted').all()
    collab_ids = [c.game_id for c in collab_entries]
    shared_games = Game.query.filter(Game.id.in_(collab_ids)).order_by(Game.created_at.desc()).all()

    def fmt(g, is_shared=False):
        return {
            'id': g.id,
            'title': g.title,
            'date': g.created_at.strftime('%Y-%m-%d'),
            'prompt': g.prompt[:70] + '...',
            'play_count': g.play_count,
            'like_count': g.like_count,
            'is_public': g.is_public,
            'is_shared': is_shared,
            'author_email': g.author.email if is_shared else None
        }

    projects_list = [fmt(g) for g in games]
    shared_list = [fmt(g, True) for g in shared_games]

    return jsonify({'projects': projects_list + shared_list})


@app.route('/api/project/<game_id>')
@login_required
def get_project(game_id):
    game = Game.query.get(game_id)
    if not game: return jsonify({'error': 'Not found'}), 404

    # Check permission (Owner or Collaborator)
    is_owner = game.user_id == current_user.id
    # Collaborator must be accepted
    collab = GameCollaborator.query.filter_by(game_id=game_id, user_id=current_user.id, status='accepted').first()

    if not is_owner and not collab:
        return jsonify({'error': 'Unauthorized'}), 403

    # === ACCESS CONTROL: CODE VIEW / EDIT ===
    # Free/Starter can PLAY (get code for iframe) but NOT view source for editing
    # Plans: 'free', 'starter' -> NO access to AI edit/source view
    can_edit_code = current_user.tier in ['premium', 'developer', 'ultra']

    # Collaborator Role Check: Admin (Collaborator) can edit content, but cannot remove users (handled in UI/Routes)
    role = 'owner' if is_owner else (collab.role if collab else 'viewer')

    # Parse settings_json if it exists
    settings_data = {}
    if game.settings_json:
        try:
            settings_data = json.loads(game.settings_json)
        except:
            settings_data = {}

    # Include is_public status and settings in response
    return jsonify({
        'code': game.code_html,  # Always sent so iframe can render
        'prompt': game.prompt,
        'title': game.title,
        'is_public': game.is_public,
        'settings': settings_data,
        'is_owner': is_owner,
        'can_edit_code': can_edit_code,  # Frontend uses this to hide Magic Edit / Visual Editor
        'role': role
    })


@app.route('/api/project/delete/<game_id>', methods=['POST'])
@login_required
def delete_project(game_id):
    # Only owner can delete
    game = Game.query.filter_by(id=game_id, user_id=current_user.id).first()
    if not game:
        return jsonify({'success': False, 'message': 'Project not found or access denied.'}), 404

    # Clean up collaborators
    GameCollaborator.query.filter_by(game_id=game_id).delete()
    # Clean up likes
    GameLike.query.filter_by(game_id=game_id).delete()

    db.session.delete(game)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Project permanently deleted.'}), 200


@app.route('/api/project/update_metadata/<game_id>', methods=['POST'])
@login_required
def update_metadata(game_id):
    # Owner or Admin(Collab) can update
    game = Game.query.get(game_id)
    if not game: return jsonify({'success': False, 'message': 'Not found'}), 404

    is_owner = game.user_id == current_user.id
    is_collab = GameCollaborator.query.filter_by(game_id=game_id, user_id=current_user.id, status='accepted').first()

    if not is_owner and not is_collab:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    data = request.json
    title = data.get('title', '').strip()
    description = data.get('description', '').strip()
    publish_identity = data.get('publish_identity', 'anon')  # 'anon' or 'username'

    # Bad Words Filter (Mandatory)
    if not check_content_safety(title) or not check_content_safety(description):
        return jsonify({'success': False, 'message': 'Content contains prohibited words.'}), 400

    if title: game.title = title
    if description: game.prompt = description  # Mapping description to prompt as requested

    # Store identity preference in settings_json
    settings = {}
    if game.settings_json:
        try:
            settings = json.loads(game.settings_json)
        except:
            settings = {}

    settings['author_display_mode'] = publish_identity
    game.settings_json = json.dumps(settings)

    db.session.commit()
    return jsonify({'success': True})


@app.route('/api/share/<game_id>')
@login_required
def share_game(game_id):
    # Visibility Change: Owner Only
    game = Game.query.get(game_id)
    if not game: return jsonify({'error': 'Not found'}), 404

    if game.user_id != current_user.id:
        return jsonify({'error': 'Only Owner can change visibility'}), 403

    if current_user.tier == 'free':
        return jsonify({'error': 'Upgrade required to share'}), 402

    game.is_public = True
    db.session.commit()
    domain = os.getenv('DOMAIN_URL', 'http://localhost:5000')
    return jsonify({'url': f"{domain}/s/{game.public_id}"})


@app.route('/api/generate', methods=['POST'])
@login_required
def generate():
    user = User.query.get(current_user.id)

    # === CRITICAL FIX: CONCURRENCY LOCK & COOLDOWN ===
    # 1. Check if user is already generating (Hard Lock)
    if user.id in GENERATION_LOCKS:
        print(f"User {user.id} hit concurrency lock. Rejected.")
        return jsonify({'error': 'ALREADY_PROCESSING', 'message_for_user': 'Generation in progress. Please wait.'}), 429

    # 2. Check Cooldown (60 seconds)
    last_time = USER_COOLDOWNS.get(user.id, 0)
    current_time = time.time()
    if current_time - last_time < 60:
        wait_time = int(60 - (current_time - last_time))
        print(f"User {user.id} hit cooldown. Wait: {wait_time}s")
        return jsonify(
            {'error': 'COOLDOWN', 'message_for_user': f'Please wait {wait_time}s before generating again.'}), 429

    # Acquire Lock
    GENERATION_LOCKS.add(user.id)

    try:
        # === START GENERATION LOGIC ===
        prompt = request.json.get('prompt')
        if not prompt: return jsonify({'error': 'Prompt is empty'}), 400

        # Bad words check on generation prompt
        if not check_content_safety(prompt):
            return jsonify({'error': 'SAFETY_BLOCK', 'message_for_user': 'Prompt contains prohibited content.'}), 400

        # FEATURE 5 & 4: Model Selection & Usage Checking
        requested_model = request.json.get('model', 'flash')  # 'flash' or 'pro'

        # Resolve Model Name
        model_api_name = GEMINI_MODEL_NAME  # Default Flash
        usage_type = 'normal'

        if requested_model == 'pro':
            # Check if tier allows pro
            if user.tier not in ['developer', 'ultra']:
                return jsonify({'error': 'PRO_MODEL_LOCKED',
                                'message_for_user': 'Gemini 2.5 Pro requires Developer or Ultra tier.'}), 403
            model_api_name = GEMINI_MODEL_PRO
            usage_type = 'pro'
        else:
            # Default Flash
            usage_type = 'normal'

        # CHECK EDIT PERMISSION (Code Access)
        # If this is an EDIT (implied if we pass context or refined prompt, though standard generate is new game)
        # The prompt implies Magic Edit hits this too.
        # We will assume this route handles NEW builds.
        # IF this were an update to an existing game ID, we'd need to check tier.
        # The frontend uses this for Magic Edit too.
        # If the user is passing "Original Idea: ..." it's likely an edit.

        # Strictly Enforce Code Access for edits or new gens if we want to restrict Free/Starter
        # Prompt Rule: "Free/Starter -> NO access to generated game SOURCE CODE"
        # This implies they can generate a game (play it), but NOT edit the code via AI.
        # If the prompt contains "CURRENT CODE", it is a Magic Edit request.
        if "CURRENT CODE" in prompt and user.tier in ['free', 'starter']:
            return jsonify({'error': 'TIER_RESTRICTION',
                            'message_for_user': 'AI Code Editing requires Premium plan or higher.'}), 403

        # Check Remaining Uses
        remaining = get_remaining_uses(user, usage_type)
        if remaining <= 0:
            reset_time = user.last_usage_reset + datetime.timedelta(days=1)
            wait_min = int((reset_time - datetime.datetime.utcnow()).total_seconds() / 60)
            hh = wait_min // 60
            mm = wait_min % 60
            msg = f"Max {usage_type.upper()} uses reached. Try again in {hh:02d}:{mm:02d}."
            return jsonify({'error': 'LIMIT_REACHED', 'message_for_user': msg}), 429

        # Legacy check for Free Tier lifetime limit
        if user.tier == 'free' and user.games_generated >= 1:
            # If it's a NEW generation (not an edit), block
            if "CURRENT CODE" not in prompt:
                return jsonify({'error': 'PAYWALL_TRIGGERED', 'tier': user.tier}), 402

        settings = request.json.get('settings')
        settings_json_str = None
        if settings:
            try:
                settings_json_str = json.dumps(settings)
            except:
                pass

        try:
            # Use the fixed, robust generation logic with selected model
            code = generate_game_logic(prompt, model_api_name)

            # Deduct Usage
            if usage_type == 'pro':
                user.daily_usage_pro += 1
            else:
                user.daily_usage_normal += 1
                user.games_generated += 1  # Keep tracking lifetime for free tier stats

            db.session.commit()

        except Exception as e:
            error_msg = f"Error doing generation: {e}"
            print(f"Server Error Log: {error_msg}")
            return jsonify(
                {'error': error_msg, 'type': 'GENERATION_ERROR', 'message_for_user': 'Error doing generation'}), 500

        # Extract title from code (simple regex for <title> content)
        match = re.search(r'<title>(.*?)</title>', code, re.IGNORECASE | re.DOTALL)
        title = match.group(1).strip() if match else prompt.split('\n')[
                                                         0].strip() or f"Project {datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        # Only create new DB entry if it's NOT an edit (Frontend handles saving edits differently or overwrites)
        # The current frontend architecture for Magic Edit generates code then returns it, client saves via specialized route or overwrites local.
        # The standard 'generate' creates a NEW game ID.
        # If "CURRENT CODE" is in prompt, it's a transient generation for the editor. We don't save to DB yet.

        if "CURRENT CODE" in prompt:
            return jsonify({'success': True, 'code': code})

        new_game = Game(user_id=user.id, title=title, prompt=prompt, code_html=code)
        if settings_json_str:
            new_game.settings_json = settings_json_str

        db.session.add(new_game)
        db.session.commit()

        # Update cooldown timestamp only on success to prevent lockout from system errors
        USER_COOLDOWNS[user.id] = time.time()

        return jsonify({'success': True, 'game_id': new_game.id, 'code': code})

    finally:
        # ALWAYS release the lock
        GENERATION_LOCKS.discard(user.id)


@app.route('/api/download/<game_id>', methods=['GET'])
@login_required
def download(game_id):
    download_type = request.args.get('type', 'html')  # New parameter: html or pwa

    if current_user.tier == 'free':
        return jsonify({'error': 'Upgrade required'}), 402

    game = Game.query.get(game_id)
    # Allow collaborators to download?
    is_owner = game.user_id == current_user.id
    is_collab = GameCollaborator.query.filter_by(game_id=game_id, user_id=current_user.id, status='accepted').first()

    if not is_owner and not is_collab:
        return jsonify({'error': 'Auth'}), 403

    # Sanitize title for filename
    title = re.sub(r'[^a-zA-Z0-9_]+', '', game.title.replace(' ', '_'))[:50] or 'AETHER_Game'
    zip_filename = f'{title}_download.zip'

    # Create zip in memory
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Main HTML File
        zf.writestr('index.html', game.code_html)
        # 2. PWA Files if requested
        if download_type == 'pwa':
            manifest_content = f"""{{
    "name": "{game.title}",
    "short_name": "{title}",
    "description": "{game.title} - A game created with AETHER.AI",
    "start_url": "./{title}.html",
    "display": "standalone",
    "background_color": "#000000",
    "theme_color": "#00f3ff",
    "icons": [
        {{ "src": "icon-512.png", "sizes": "512x512", "type": "image/png" }}
    ]
}}"""
            zf.writestr('manifest.json', manifest_content)

            service_worker_content = f"""
const CACHE_NAME = 'aether-game-v1';
const urlsToCache = [
    './{title}.html',
    './manifest.json'
];

self.addEventListener('install', event => {{
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {{
                return cache.addAll(urlsToCache);
            }})
    );
}});

self.addEventListener('fetch', event => {{
    event.respondWith(
        caches.match(event.request)
            .then(response => {{
                return response || fetch(event.request);
            }})
    );
}});
"""
            zf.writestr('service-worker.js', service_worker_content)
            mock_icon = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0BMVEX/AAAAAwcHBwD8qI8AAAAJcEhZcwAAIDoAAAByAAAEsQByAAAEsQByAAAEsQAAP0+9yAAAAFE5JREFUCB1jjmCAAAADwAAAAQABF3uJPAAAAAElFTkSuQmCC"
            zf.writestr('icon-512.png', mock_icon)

        readme_content = f"Your game '{game.title}' was generated by AETHER.AI.\n\nTo play, open '{title}.html' in your browser.\n\n{'This is a PWA package. Upload all files to a web server to install the PWA on mobile devices.' if download_type == 'pwa' else ''}"
        zf.writestr('README.txt', readme_content)

    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_filename
    )


@app.route('/api/checkout', methods=['POST'])
@login_required
def checkout():
    tier = request.json.get('tier')
    interval = request.json.get('interval', 'month')  # Feature 7
    domain = os.getenv('DOMAIN_URL', 'http://localhost:5000')

    # --- TIER ENFORCEMENT & UPGRADE LOGIC ---
    current_level = TIER_LEVELS.get(current_user.tier, 0)
    requested_level = TIER_LEVELS.get(tier, 0)

    if requested_level <= current_level:
        return jsonify({
            'error': 'Invalid Upgrade',
            'message': f'You are already on {current_user.tier.upper()}. You can only upgrade to higher tiers.'
        }), 400

    # --- REAL STRIPE CHECKOUT ---
    price_map = {
        'starter': {'month': 'price_starter_mo', 'year': 'price_starter_yr'},
        'developer': {'month': 'price_dev_mo', 'year': 'price_dev_yr'},
        'ultra': {'month': 'price_ultra_mo', 'year': 'price_ultra_yr'},
    }

    # Logic to handle mock prices if real IDs aren't in env (Fallback for strict generation context)
    # But prompt says "REAL ONLY".
    # If standard Stripe keys aren't set, we return error.
    if not stripe.api_key:
        return jsonify({'error': 'Configuration Error', 'message': 'Payment system not configured.'}), 500

    # Use a generic product creation if specific price IDs aren't hardcoded or in ENV
    # For robustness, we'll create a session with line_items constructed ad-hoc if possible,
    # but subscription mode usually requires Price IDs.
    # We will assume a 'setup mode' or just generate a session with Amount for One-Time
    # OR create a Price object on the fly (less ideal).
    # BEST PRACTICE: Assume ENV variables exist for prices or use lookup.
    # FALLBACK: Create Session with price_data (creates product on fly)

    amounts = {
        'starter': {'month': 500, 'year': 3600},
        'developer': {'month': 2000, 'year': 14400},
        'ultra': {'month': 8200, 'year': 58800},
    }

    amount = amounts.get(tier, {}).get(interval)
    if not amount: return jsonify({'error': 'Invalid Tier'}), 400

    try:
        checkout_session = stripe.checkout.Session.create(
            client_reference_id=f"{current_user.id}:{tier}",
            payment_method_types=['card'],  # PayPal often requires specific config in Stripe dashboard
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': f"AETHER {tier.upper()} ({interval})",
                    },
                    'unit_amount': amount,
                    'recurring': {'interval': interval}
                },
                'quantity': 1,
            }],
            mode='subscription',
            success_url=domain + '/?mock_success=true',  # Redirect to home with success flag
            cancel_url=domain + '/',
        )
        return jsonify({'success': True, 'url': checkout_session.url})
    except Exception as e:
        print(f"Stripe Error: {e}")
        return jsonify({'error': 'Payment Init Failed', 'message': str(e)}), 500


@app.route('/webhook', methods=['POST'])
def webhook():
    # Stripe
    payload = request.data
    sig = request.headers.get('Stripe-Signature')
    endpoint_secret = os.getenv('STRIPE_WEBHOOK_SECRET')

    if not endpoint_secret: return 'Webhook secret not configured', 500

    try:
        event = stripe.Webhook.construct_event(payload, sig, endpoint_secret)
    except ValueError as e:
        return 'Invalid payload', 400
    except stripe.error.SignatureVerificationError as e:
        return 'Invalid signature', 400

    if event['type'] in ['checkout.session.completed', 'invoice.payment.succeeded']:
        session = event['data']['object']
        ref = session.get('client_reference_id')

        if ref:
            uid, tier = ref.split(':')
            user = User.query.get(int(uid))
            if user:
                user.tier = tier
                user.games_generated = 0
                db.session.commit()
                print(f"User {uid} upgraded to {tier} via webhook.")

    return 'OK', 200


# === AETHER FEATURE ADDITION START ===

@app.route('/api/gallery')
def gallery_list():
    # Feature 2: Public Gallery Listing
    games = Game.query.filter_by(is_public=True).order_by(Game.created_at.desc()).limit(50).all()

    gallery_data = []
    for g in games:
        # Determine Display Name based on preferences
        user_display = "Anonymous"

        # Check stored preference
        settings = {}
        if g.settings_json:
            try:
                settings = json.loads(g.settings_json)
            except:
                settings = {}

        mode = settings.get('author_display_mode', 'anon')
        if mode == 'username':
            email_part = g.author.email.split('@')[0]
            if len(email_part) > 3:
                user_display = email_part[:3] + "***"
            else:
                user_display = email_part

        gallery_data.append({
            'id': g.id,
            'public_id': g.public_id,
            'title': g.title,
            'author': user_display,
            'play_count': g.play_count,
            'like_count': g.like_count,
            'prompt_snippet': g.prompt[:60] + "..." if g.prompt else "No description"
        })
    return jsonify({'games': gallery_data})


@app.route('/api/game/<game_id>/play', methods=['POST'])
def record_play(game_id):
    # Feature 3: Increment Play Count
    game = Game.query.get_or_404(game_id)
    game.play_count += 1
    db.session.commit()
    return jsonify({'success': True, 'play_count': game.play_count})


@app.route('/api/game/<game_id>/like', methods=['POST'])
@login_required
def record_like(game_id):
    # Feature 3: Toggle Like with Backend Enforcement
    game = Game.query.get_or_404(game_id)
    existing_like = GameLike.query.filter_by(user_id=current_user.id, game_id=game_id).first()

    if existing_like:
        # UNLIKE
        db.session.delete(existing_like)
        game.like_count = max(0, game.like_count - 1)
        liked = False
    else:
        # LIKE
        new_like = GameLike(user_id=current_user.id, game_id=game_id)
        db.session.add(new_like)
        game.like_count += 1
        liked = True

    db.session.commit()
    return jsonify({'success': True, 'like_count': game.like_count, 'liked': liked})


@app.route('/api/game/public/<game_id>', methods=['GET'])
def get_public_game_code(game_id):
    # Helper to load a public game into the builder iframe
    game = Game.query.get_or_404(game_id)
    if not game.is_public and (not current_user.is_authenticated or game.user_id != current_user.id):
        # Check collaborator
        if current_user.is_authenticated:
            is_collab = GameCollaborator.query.filter_by(game_id=game_id, user_id=current_user.id,
                                                         status='accepted').first()
            if is_collab:
                return jsonify({'code': game.code_html, 'title': game.title, 'id': game.id})
        return jsonify({'error': 'Private Game'}), 403
    return jsonify({'code': game.code_html, 'title': game.title, 'id': game.id})


# --- NEW ROUTE: PUBLIC TOGGLE ---
@app.route('/api/project/toggle_public/<game_id>', methods=['POST'])
@login_required
def toggle_public(game_id):
    # Owner ONLY
    game = Game.query.filter_by(id=game_id, user_id=current_user.id).first()
    if not game:
        return jsonify({'success': False, 'message': 'Project not found or unauthorized.'}), 404

    game.is_public = not game.is_public
    db.session.commit()
    return jsonify({'success': True, 'is_public': game.is_public})


# --- VISUAL EDITOR SAVE ROUTE ---
@app.route('/api/game/<game_id>/save_visuals', methods=['POST'])
@login_required
def save_visuals(game_id):
    # TIER CHECK: Free/Starter cannot save visual edits
    if current_user.tier in ['free', 'starter']:
        return jsonify({'success': False, 'message': 'Upgrade required to save edits.'}), 402

    # Check permissions (Owner OR Collaborator)
    game = Game.query.get(game_id)
    if not game: return jsonify({'success': False, 'message': 'Project not found.'}), 404

    is_owner = game.user_id == current_user.id
    is_collab = GameCollaborator.query.filter_by(game_id=game_id, user_id=current_user.id, status='accepted').first()

    if not is_owner and not is_collab:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    data = request.json
    visual_data = data.get('visualData')
    global_settings = data.get('globalSettings')

    # Load existing settings or init
    settings = {}
    if game.settings_json:
        try:
            settings = json.loads(game.settings_json)
        except:
            settings = {}

    # Update nested visual overrides
    settings['visual_overrides'] = visual_data
    # Merge global settings carefully (game width etc)
    if 'global_settings' not in settings: settings['global_settings'] = {}
    if global_settings:
        settings['global_settings'].update(global_settings)

    # Save back
    game.settings_json = json.dumps(settings)
    db.session.commit()

    return jsonify({'success': True})


# --- TEAM COLLABORATION ROUTES (Feature 6 & BUG FIX) ---

@app.route('/api/game/<game_id>/invite', methods=['POST'])
@login_required
def invite_collaborator(game_id):
    # Only Owner can invite
    game = Game.query.filter_by(id=game_id, user_id=current_user.id).first()
    if not game: return jsonify({'error': 'Not found or unauthorized'}), 403

    email = request.json.get('email')
    if not email: return jsonify({'error': 'Email required'}), 400

    user_to_invite = User.query.filter_by(email=email).first()
    if not user_to_invite:
        return jsonify({'error': 'User must have an AETHER account first.'}), 404

    if user_to_invite.id == current_user.id:
        return jsonify({'error': 'You cannot invite yourself.'}), 400

    existing = GameCollaborator.query.filter_by(game_id=game_id, user_id=user_to_invite.id).first()
    if existing:
        return jsonify({'error': 'User already invited.'}), 400

    # Create Pending Invite
    collab = GameCollaborator(game_id=game_id, user_id=user_to_invite.id, status='pending', role='admin')
    db.session.add(collab)
    db.session.commit()

    return jsonify({'success': True, 'message': f'{email} invited.'})


@app.route('/api/user/invites', methods=['GET'])
@login_required
def get_invites():
    # List pending invites for the current user
    invites = GameCollaborator.query.filter_by(user_id=current_user.id, status='pending').all()
    result = []
    for inv in invites:
        game = Game.query.get(inv.game_id)
        if game:
            result.append({
                'id': inv.id,
                'game_title': game.title,
                'owner_email': game.author.email
            })
    return jsonify({'invites': result})


@app.route('/api/user/invite/respond', methods=['POST'])
@login_required
def respond_invite():
    data = request.json
    invite_id = data.get('invite_id')
    action = data.get('action')  # 'accept' or 'deny'

    invite = GameCollaborator.query.filter_by(id=invite_id, user_id=current_user.id, status='pending').first()
    if not invite: return jsonify({'error': 'Invite not found'}), 404

    if action == 'accept':
        invite.status = 'accepted'
        db.session.commit()
        return jsonify({'success': True, 'message': 'Joined team.'})
    elif action == 'deny':
        db.session.delete(invite)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Invite denied.'})

    return jsonify({'error': 'Invalid action'}), 400


@app.route('/api/game/<game_id>/collaborators', methods=['GET'])
@login_required
def get_collaborators(game_id):
    # Owner only
    game = Game.query.filter_by(id=game_id, user_id=current_user.id).first()
    if not game: return jsonify({'error': 'Unauthorized'}), 403

    collabs = GameCollaborator.query.filter_by(game_id=game_id).all()
    result = []
    for c in collabs:
        u = User.query.get(c.user_id)
        result.append({
            'id': c.id,
            'email': u.email,
            'added_at': c.added_at.strftime('%Y-%m-%d'),
            'status': c.status,
            'role': c.role
        })

    return jsonify({'collaborators': result})


@app.route('/api/game/<game_id>/remove_collaborator', methods=['POST'])
@login_required
def remove_collaborator(game_id):
    # Owner only
    game = Game.query.filter_by(id=game_id, user_id=current_user.id).first()
    if not game: return jsonify({'error': 'Unauthorized'}), 403

    collab_id = request.json.get('collab_id')
    GameCollaborator.query.filter_by(id=collab_id, game_id=game_id).delete()
    db.session.commit()

    return jsonify({'success': True})


@app.route('/api/contact', methods=['POST'])
def contact_form():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    message = data.get('message')

    if not all([name, email, message]):
        return jsonify({'success': False, 'message': 'All fields required'}), 400

    if not check_content_safety(message):
        return jsonify({'success': False, 'message': 'Invalid content'}), 400

    # Send email to admin
    admin_email = os.getenv('ADMIN_EMAIL', 'support@aether.ai')
    subject = f"Contact Form: {name}"
    body = f"From: {name} ({email})\n\n{message}"

    send_email(admin_email, subject, body)
    return jsonify({'success': True})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


