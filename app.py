from pyexpat import features

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import json
from datetime import datetime
import os
from datetime import timedelta
from flask_socketio import SocketIO, emit

# Line 13 can be blank

# Load models once at startup
import tensorflow as tf
isolation_forest = joblib.load("models/isolation_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
autoencoder = tf.keras.models.load_model("models/autoencoder.keras")
ae_threshold = joblib.load("models/autoencoder_threshold.pkl")
print("All models loaded successfully!")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
app.secret_key = "fyp_anomaly_detection_secret_key_2025_fixed"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

db = SQLAlchemy(app)

# ----------------------------------------
# DATABASE MODELS
# ----------------------------------------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'student' or 'examiner'
    full_name = db.Column(db.String(100), nullable=False)

class Exam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    duration = db.Column(db.Integer, default=30)  # minutes
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey("exam.id"), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    option_a = db.Column(db.String(200), nullable=False)
    option_b = db.Column(db.String(200), nullable=False)
    option_c = db.Column(db.String(200), nullable=False)
    option_d = db.Column(db.String(200), nullable=False)
    correct_answer = db.Column(db.String(1), nullable=False)

class ExamSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    exam_id = db.Column(db.Integer, db.ForeignKey("exam.id"), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    score = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), default="in_progress")

class BehaviourLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("exam_session.id"), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    tab_switches = db.Column(db.Integer, default=0)
    total_clicks = db.Column(db.Integer, default=0)
    time_on_exam = db.Column(db.Float, default=0)
    avg_time_per_question = db.Column(db.Float, default=0)
    anomaly_label = db.Column(db.Integer, nullable=True)
    anomaly_score = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    anomaly_reason = db.Column(db.String(500), nullable=True)
    ae_label = db.Column(db.Integer, nullable=True)
    ae_score = db.Column(db.Float, nullable=True)
    ae_threshold = db.Column(db.Float, nullable=True)
    alert_sent = db.Column(db.Boolean, default=False)

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    session_id = db.Column(db.Integer, db.ForeignKey("exam_session.id"))
    message = db.Column(db.String(500))
    severity = db.Column(db.String(20))
    is_read = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ----------------------------------------
# ROUTES — AUTH
# ----------------------------------------

@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["role"] = user.role
            session.permanent = True
            session["full_name"] = user.full_name
            if user.role == "student":
                return redirect(url_for("student_dashboard"))
            else:
                return redirect(url_for("examiner_dashboard"))
        return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ----------------------------------------
# ROUTES — STUDENT
# ----------------------------------------

@app.route("/student")
def student_dashboard():
    if session.get("role") != "student":
        return redirect(url_for("login"))
    exams = Exam.query.all()
    return render_template("student_dashboard.html", 
                         exams=exams, 
                         full_name=session.get("full_name"))

@app.route("/exam/<int:exam_id>")
def take_exam(exam_id):
    if session.get("role") != "student":
        return redirect(url_for("login"))
    exam = Exam.query.get_or_404(exam_id)
    import random
    questions = Question.query.filter_by(exam_id=exam_id).all()
    random.shuffle(questions)
    exam_session = ExamSession(
        student_id=session["user_id"],
        exam_id=exam_id
    )
    db.session.add(exam_session)
    db.session.commit()
    session["exam_session_id"] = exam_session.id
    return render_template("exam.html", exam=exam, questions=questions)

@app.route("/submit_exam", methods=["POST"])
def submit_exam():
    if not session.get("user_id"):
        return jsonify({"success": False, "error": "Not logged in"}), 401

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data received"}), 400

    session_id = session.get("exam_session_id")
    exam_session = ExamSession.query.get(session_id)
    score = 0

    if exam_session:
        questions = Question.query.filter_by(exam_id=exam_session.exam_id).all()
        correct = 0
        answers = data.get("answers", {})
        for q in questions:
            if answers.get(str(q.id)) == q.correct_answer:
                correct += 1
        score = (correct / len(questions)) * 100 if questions else 0

        exam_session.end_time = datetime.utcnow()
        exam_session.score = score
        exam_session.status = "completed"
        db.session.commit()

        behaviour = data.get("behaviour", {})
        time_on_exam = behaviour.get("time_on_exam", 0)
        total_clicks = behaviour.get("total_clicks", 0)
        tab_switches = behaviour.get("tab_switches", 0)
        avg_time_per_question = time_on_exam / len(questions) if questions else 0
        activity_rate = total_clicks / max(time_on_exam, 1)

        log = BehaviourLog(
            session_id=session_id,
            student_id=session.get("user_id"),
            tab_switches=tab_switches,
            total_clicks=total_clicks,
            time_on_exam=time_on_exam,
            avg_time_per_question=avg_time_per_question,
            anomaly_label=1,
            anomaly_score=0.0,
            ae_label=1,
            ae_score=0.0,
            ae_threshold=0.0
        )
        db.session.add(log)
        db.session.commit()

        try:
            features = np.array([[
                score, total_clicks, time_on_exam,
                tab_switches, avg_time_per_question, activity_rate
            ]])
            features_scaled = scaler.transform(features)

            if_prediction = isolation_forest.predict(features_scaled)
            if_score = isolation_forest.decision_function(features_scaled)
            if_label = int(if_prediction[0])

            reconstruction = autoencoder.predict(features_scaled, verbose=0)
            ae_error = float(np.mean(np.power(features_scaled - reconstruction, 2)))
            ae_label = -1 if ae_error > ae_threshold else 1

            log.anomaly_label = if_label
            log.anomaly_score = float(if_score[0])
            log.ae_label = ae_label
            log.ae_score = ae_error
            log.ae_threshold = float(ae_threshold)
            if if_label == -1 or ae_label == -1:

                student = User.query.get(session.get("user_id"))

                notification = Notification(
                    student_id=session.get("user_id"),
                    session_id=session_id,
                    severity="HIGH",
                    message=f"Student {student.full_name} was flagged for anomalous behaviour."
    )

                db.session.add(notification)
            db.session.commit()

        except Exception as e:
            print(f"Anomaly detection error: {e}")

    return jsonify({"success": True, "score": score, "session_id": session_id})

# ----------------------------------------
# ROUTES — EXAMINER
# ----------------------------------------

@app.route("/examiner")
def examiner_dashboard():
    if session.get("role") != "examiner":
        return redirect(url_for("login"))
    return render_template("examiner.html", full_name=session.get("full_name"))

@app.route("/api/live_students")
def live_students():
    if session.get("role") != "examiner":
        return jsonify([])
    
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(hours=2)
    
    active_sessions = ExamSession.query.filter_by(status="in_progress").filter(
        ExamSession.start_time >= cutoff
    ).all()
    
    students = []
    for s in active_sessions:
        student = User.query.get(s.student_id)
        exam = Exam.query.get(s.exam_id)
        log = BehaviourLog.query.filter_by(session_id=s.id).first()
        students.append({
            "name": student.full_name,
            "exam": exam.title,
            "start_time": s.start_time.strftime("%H:%M:%S"),
            "tab_switches": log.tab_switches if log else 0,
            "total_clicks": log.total_clicks if log else 0,
            "anomaly_label": log.anomaly_label if log else None,
            "anomaly_score": log.anomaly_score if log else None,
            "status": "⚠️ Suspicious" if log and log.anomaly_label == -1 else "✅ Normal"
        })
    return jsonify(students)

@app.route("/api/behaviour_logs")
def behaviour_logs():
    if session.get("role") != "examiner":
        return jsonify([])
    logs = BehaviourLog.query.order_by(BehaviourLog.timestamp.desc()).limit(50).all()
    result = []
    for log in logs:
        student = User.query.get(log.student_id)
        result.append({
            "name": student.full_name,
            "tab_switches": log.tab_switches,
            "total_clicks": log.total_clicks,
            "time_on_exam": round(log.time_on_exam, 2),
            "anomaly_label": log.anomaly_label,
            "anomaly_score": round(log.anomaly_score, 4) if log.anomaly_score else None,
            "status": "⚠️ Anomalous" if log.anomaly_label == -1 else "✅ Normal",
            "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        })
    return jsonify(result)

@app.route("/api/notifications")
def notifications():

    if session.get("role") != "examiner":
        return jsonify([])

    notifications = Notification.query.order_by(
        Notification.timestamp.desc()
    ).limit(20).all()

    results = []

    for n in notifications:

        student = User.query.get(n.student_id)

        results.append({
            "id": n.id,
            "student": student.full_name,
            "message": n.message,
            "severity": n.severity,
            "timestamp": n.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "is_read": n.is_read
        })

    return jsonify(results)

# ----------------------------------------
# ROUTES — EXAM MANAGEMENT
# ----------------------------------------

@app.route("/examiner/exams")
def manage_exams():
    if session.get("role") != "examiner":
        return redirect(url_for("login"))
    exams = Exam.query.all()
    return render_template("manage_exams.html", 
                         exams=exams, 
                         full_name=session.get("full_name"))

@app.route("/examiner/exams/create", methods=["GET", "POST"])
def create_exam():
    if session.get("role") != "examiner":
        return redirect(url_for("login"))
    if request.method == "POST":
        title = request.form.get("title")
        duration = request.form.get("duration", 30)
        exam = Exam(title=title, duration=int(duration))
        db.session.add(exam)
        db.session.commit()
        return redirect(url_for("add_question", exam_id=exam.id))
    return render_template("create_exam.html", full_name=session.get("full_name"))

@app.route("/examiner/exams/<int:exam_id>/questions", methods=["GET", "POST"])
def add_question(exam_id):
    if session.get("role") != "examiner":
        return redirect(url_for("login"))
    exam = Exam.query.get_or_404(exam_id)
    if request.method == "POST":
        question = Question(
            exam_id=exam_id,
            question_text=request.form.get("question_text"),
            option_a=request.form.get("option_a"),
            option_b=request.form.get("option_b"),
            option_c=request.form.get("option_c"),
            option_d=request.form.get("option_d"),
            correct_answer=request.form.get("correct_answer")
        )
        db.session.add(question)
        db.session.commit()
        if request.form.get("action") == "save_and_finish":
            return redirect(url_for("manage_exams"))
        return redirect(url_for("add_question", exam_id=exam_id))
    questions = Question.query.filter_by(exam_id=exam_id).all()
    return render_template("add_question.html", 
                         exam=exam, 
                         questions=questions,
                         full_name=session.get("full_name"))

@app.route("/examiner/exams/<int:exam_id>/delete", methods=["POST"])
def delete_exam(exam_id):
    if session.get("role") != "examiner":
        return redirect(url_for("login"))
    Question.query.filter_by(exam_id=exam_id).delete()
    ExamSession.query.filter_by(exam_id=exam_id).delete()
    Exam.query.filter_by(id=exam_id).delete()
    db.session.commit()
    return redirect(url_for("manage_exams"))

@app.route("/results/<int:session_id>")
def results(session_id):
    if session.get("role") != "student":
        return redirect(url_for("login"))
    exam_session = ExamSession.query.get_or_404(session_id)
    student = User.query.get(exam_session.student_id)
    exam = Exam.query.get(exam_session.exam_id)
    log = BehaviourLog.query.filter_by(session_id=session_id).first()
    return render_template("results.html",
                         exam_session=exam_session,
                         student=student,
                         exam=exam,
                         log=log)
@app.route("/api/update_behaviour", methods=["POST"])
def update_behaviour():
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    session_id = session.get("exam_session_id")
    
    if not session_id:
        return jsonify({"error": "No active session"}), 400
    
    log = BehaviourLog.query.filter_by(session_id=session_id).first()
    
    if not log:
        log = BehaviourLog(
            session_id=session_id,
            student_id=session["user_id"],
            tab_switches=data.get("tab_switches", 0),
            total_clicks=data.get("total_clicks", 0),
            time_on_exam=data.get("time_on_exam", 0),
            avg_time_per_question=data.get("avg_time_per_question", 0)
        )
        db.session.add(log)
    else:
        log.tab_switches = data.get("tab_switches", 0)
        log.total_clicks = data.get("total_clicks", 0)
        log.time_on_exam = data.get("time_on_exam", 0)
        log.avg_time_per_question = data.get("avg_time_per_question", 0)
    activity_rate = log.total_clicks / max(log.time_on_exam, 1)

    features = np.array([[
    0,  # score unavailable during exam
    log.total_clicks,
    log.time_on_exam,
    log.tab_switches,
    log.avg_time_per_question,
    activity_rate
]])
    features_scaled = scaler.transform(features)

    if_prediction = isolation_forest.predict(features_scaled)

    if if_prediction[0] == -1:

        student = User.query.get(session["user_id"])

        socketio.emit(
            "anomaly_alert",
            {
                "student": student.full_name,
                "tab_switches": log.tab_switches,
                "clicks": log.total_clicks,
                "time": datetime.utcnow().strftime("%H:%M:%S")
            }
        )
        log.alert_sent = True
    
    db.session.commit()
    return jsonify({"success": True})

@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')

@app.route('/evaluation_data')
def evaluation_data():
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
    from sklearn.metrics import silhouette_score

    np.random.seed(42)
    n_normal, n_anomalous = 1000, 100

    normal_data = {"score":np.random.normal(65,15,n_normal).clip(0,100),"total_clicks":np.random.normal(80,20,n_normal).clip(10,200),"time_on_exam":np.random.normal(1200,300,n_normal).clip(300,1800),"tab_switches":np.random.normal(1,1,n_normal).clip(0,5),"avg_time_per_question":np.random.normal(240,60,n_normal).clip(30,600),"activity_rate":np.random.normal(0.07,0.02,n_normal).clip(0.01,0.2)}
    anomalous_data = {"score":np.random.normal(85,5,n_anomalous).clip(0,100),"total_clicks":np.random.normal(200,50,n_anomalous).clip(50,500),"time_on_exam":np.random.normal(300,100,n_anomalous).clip(60,600),"tab_switches":np.random.normal(10,3,n_anomalous).clip(5,30),"avg_time_per_question":np.random.normal(60,20,n_anomalous).clip(10,120),"activity_rate":np.random.normal(0.5,0.1,n_anomalous).clip(0.2,1.0)}

    df = pd.concat([pd.DataFrame(normal_data), pd.DataFrame(anomalous_data)], ignore_index=True)
    y_true = np.array([1]*n_normal + [-1]*n_anomalous)
    y_true_bin = (y_true == -1).astype(int)
    X_scaled = scaler.transform(df)

    if_preds = isolation_forest.predict(X_scaled)
    if_scores = isolation_forest.decision_function(X_scaled)
    if_pred_bin = (if_preds == -1).astype(int)
    if_fpr, if_tpr, _ = roc_curve(y_true_bin, -if_scores)

    reconstructions = autoencoder.predict(X_scaled, verbose=0)
    ae_errors = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
    ae_pred_bin = (ae_errors > ae_threshold).astype(int)
    ae_fpr, ae_tpr, _ = roc_curve(y_true_bin, ae_errors)

    idx = np.random.choice(len(if_scores), 200, replace=False)

    return jsonify({
        "if_precision": round(precision_score(y_true_bin,if_pred_bin,zero_division=0),4),
        "if_recall": round(recall_score(y_true_bin,if_pred_bin,zero_division=0),4),
        "if_f1": round(f1_score(y_true_bin,if_pred_bin,zero_division=0),4),
        "if_auc": round(auc(if_fpr,if_tpr),4),
        "if_sil": round(float(silhouette_score(X_scaled,if_pred_bin)),4),
        "ae_precision": round(precision_score(y_true_bin,ae_pred_bin,zero_division=0),4),
        "ae_recall": round(recall_score(y_true_bin,ae_pred_bin,zero_division=0),4),
        "ae_f1": round(f1_score(y_true_bin,ae_pred_bin,zero_division=0),4),
        "ae_auc": round(auc(ae_fpr,ae_tpr),4),
        "ae_sil": round(float(silhouette_score(X_scaled,ae_pred_bin)),4),
        "if_cm": confusion_matrix(y_true_bin,if_pred_bin).tolist(),
        "ae_cm": confusion_matrix(y_true_bin,ae_pred_bin).tolist(),
        "if_roc": {"fpr":if_fpr.tolist(),"tpr":if_tpr.tolist()},
        "ae_roc": {"fpr":ae_fpr.tolist(),"tpr":ae_tpr.tolist()},
        "if_normal_scores": (-if_scores[y_true==1]).tolist()[:200],
        "if_anom_scores": (-if_scores[y_true==-1]).tolist(),
        "ae_normal_errors": ae_errors[y_true==1].tolist()[:200],
        "ae_anom_errors": ae_errors[y_true==-1].tolist(),
        "scatter": [{"if_score":round(float(-if_scores[i]),4),"ae_error":round(float(ae_errors[i]),4),"true":int(y_true[i])} for i in idx],
        "n_anomalous": int(n_anomalous),
    })

# ----------------------------------------
#SEED DATABASE
# ----------------------------------------

def seed_database():
    # Create users
    users = [
    User(username="student1", password=generate_password_hash("pass123"), role="student", full_name="Archibong Orok"),
    User(username="student2", password=generate_password_hash("pass123"), role="student", full_name="Akinlade David"),
    User(username="student3", password=generate_password_hash("pass123"), role="student", full_name="Egbo Ugochukwu"),
    User(username="student4", password=generate_password_hash("pass123"), role="student", full_name="Okwuego Kamdy"),
    User(username="examiner1", password=generate_password_hash("exam123"), role="examiner", full_name="Dr. Smith"),
]
    for u in users:
        if not User.query.filter_by(username=u.username).first():
            db.session.add(u)

    # Create exam
    if not Exam.query.first():
        exam = Exam(title="Computer Science Fundamentals", duration=30)
        db.session.add(exam)
        db.session.commit()

        # Create questions
        questions = [
            Question(exam_id=exam.id, question_text="What does CPU stand for?",
                option_a="Central Processing Unit", option_b="Central Program Unit",
                option_c="Computer Processing Unit", option_d="Central Processor Utility",
                correct_answer="A"),
            Question(exam_id=exam.id, question_text="Which data structure uses LIFO?",
                option_a="Queue", option_b="Stack", option_c="Array", option_d="Tree",
                correct_answer="B"),
            Question(exam_id=exam.id, question_text="What is the time complexity of binary search?",
                option_a="O(n)", option_b="O(n²)", option_c="O(log n)", option_d="O(1)",
                correct_answer="C"),
            Question(exam_id=exam.id, question_text="Which language is used for web styling?",
                option_a="Python", option_b="Java", option_c="CSS", option_d="C++",
                correct_answer="C"),
            Question(exam_id=exam.id, question_text="What does RAM stand for?",
                option_a="Read Access Memory", option_b="Random Access Memory",
                option_c="Rapid Access Module", option_d="Read And Memorize",
                correct_answer="B"),
        ]
        for q in questions:
            db.session.add(q)

    db.session.commit()
    print("Database seeded successfully!")

# ----------------------------------------
# RUN
# ----------------------------------------

if __name__ == '__main__':
    # 1. Push the application context so Flask knows what app we are working with
    with app.app_context():
        print("Checking database and creating tables...")
        
        # 2. This physically creates 'database.db' and the 'behaviour_log' table
        db.create_all() 
        
        # 3. Optional: If you have a seeding function like seed_database() from your first screenshot,
        # call it here safely inside the context so your initial data is populated.
        try:
            # Uncomment the line below if you want to run your seed function automatically on startup
            #seed_database() 
            print("Database initialized successfully.")
        except Exception as e:
            print(f"Note during seeding (might already be seeded): {e}")

    print("Starting ExamGuard Server...")
    socketio.run(app, debug=True)