# ================================================================
# ScamShield AI — SMS Scam & Spam Detector (India)
#
# Lightweight, explainable, runs on any laptop. No API keys.
#
# Components:
#   1. ML classifier  : TF-IDF + Logistic Regression
#                       (UCI SMS Spam Collection + Indian scam samples)
#   2. Rule engine    : Indian scam patterns (KYC, OTP, UPI, digital
#                       arrest, electricity, jobs, courier, loans ...)
#   3. Link analysis  : shorteners, fake brand domains, risky TLDs, APKs
#   4. Sender check   : personal mobile number vs registered header ID
#   5. RAG guidance   : TF-IDF retrieval from an official-advice
#                       knowledge base
#   6. Explainable report + recommended actions
# ================================================================

import os
import re
import html
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import gradio as gr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# ================================================================
# STAGE 1 — TRAINING DATA
# ================================================================

# Built-in Indian examples. They teach the model local scam styles
# that the (UK-based) UCI dataset does not contain.
SEED_SPAM = [
    "Dear customer, your SBI account will be blocked today. Update your KYC immediately: http://sbi-kyc-update.xyz",
    "Your electricity connection will be disconnected tonight at 9:30 PM as your last month bill was not updated. Call electricity officer 9876543210 immediately.",
    "Congratulations! You have won Rs 25,00,000 in KBC lucky draw. To claim, WhatsApp 9123456780 with your bank details.",
    "Your parcel could not be delivered due to incomplete address. Update within 24 hours: bit.ly/indiapost-upd",
    "Earn Rs 5000 daily from home! Just like YouTube videos. Part time job, join on Telegram now.",
    "Your PAN card is not linked. Your account will be suspended. Click here to update: http://pan-update-gov.top",
    "You have received a refund of Rs 4,999. Enter your UPI PIN to receive the money in your account.",
    "This is CBI. A parcel with drugs was found in your name. You are under digital arrest. Do not tell anyone and call us now.",
    "Your SIM will be blocked within 2 hours by TRAI. Call 8899776655 to avoid disconnection.",
    "Pre-approved instant loan of Rs 2,00,000 at 0% interest. No documents needed. Apply now: http://quickloan.click",
    "Dear user, please share the OTP sent to your mobile to complete your reward points redemption.",
    "Your HDFC credit card reward points worth Rs 7,850 expire today. Redeem now: http://hdfc-rewards.live",
    "Install AnyDesk app so our support executive can help you get your refund processed.",
    "Your Income Tax refund of Rs 15,490 is approved. Verify your account at http://incometax-refund.online",
    "Last chance! Your Amazon account has been locked due to suspicious activity. Verify now: amzn-secure-login.site",
    "FREE recharge of Rs 299 for all Jio users. Click and claim before midnight: tinyurl.com/jio-free299",
    "Hi, I am HR from a reputed company. Earn 3000-8000 per day with simple online tasks. Reply YES on WhatsApp.",
    "Your EPF claim is pending. Download the app to update details: http://epfo-update.in/app.apk",
]

SEED_HAM = [
    "Your OTP for login is 482913. Do not share it with anyone. -SBI",
    "Rs 1,250 debited from A/c XX1234 on 24-11-26 via UPI. If not done by you, call the official customer care.",
    "Hey, are we meeting at 6 for dinner?",
    "Mom, I reached the hostel safely. Will call you at night.",
    "Your Amazon order has been shipped and will arrive by Friday.",
    "Class is cancelled tomorrow, assignment deadline extended to Monday.",
    "Your electricity bill of Rs 1,340 is generated. Pay by 5 Dec on the official app.",
    "Happy birthday bro! Party tonight?",
    "Your appointment with Dr. Sharma is confirmed for 10 AM tomorrow.",
    "Can you send me the notes of today's lecture?",
    "Your Jio recharge of Rs 299 is successful. Validity 28 days.",
    "Meeting moved to 3 PM, please inform the team.",
    "Your parcel has been delivered. Thank you for shopping with Flipkart.",
    "I'll call you after the exam, ok?",
    "Lab practical marks are uploaded on the college portal.",
    "Bhaiya, please bring milk and bread while coming home.",
]


def load_dataset():
    """
    Load the UCI SMS Spam Collection if it is in the project folder.
    Supports:
      - spam.csv        (Kaggle version, columns v1/v2, latin-1)
      - SMSSpamCollection (original UCI tab-separated file)
    """
    df, source = None, "Built-in sample messages only"

    try:
        if os.path.exists("spam.csv"):
            df = pd.read_csv("spam.csv", encoding="latin-1")
            df = df.iloc[:, :2]
            df.columns = ["label", "text"]
            source = "spam.csv — UCI SMS Spam Collection (Kaggle)"
        elif os.path.exists("SMSSpamCollection"):
            df = pd.read_csv(
                "SMSSpamCollection", sep="\t", header=None,
                names=["label", "text"], quoting=3, encoding="utf-8"
            )
            source = "SMSSpamCollection — UCI SMS Spam Collection"
    except Exception as e:
        print(f"Could not read dataset: {e}")
        df = None

    if df is None:
        return [], [], source

    df = df.dropna()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(
        {"spam": 1, "ham": 0}
    )
    df = df.dropna(subset=["label"])

    return df["text"].astype(str).tolist(), df["label"].astype(int).tolist(), source


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=10,
        )),
    ])


print("Loading data and training model...")

ds_texts, ds_labels, DATA_SOURCE = load_dataset()
seed_texts = SEED_SPAM + SEED_HAM
seed_labels = [1] * len(SEED_SPAM) + [0] * len(SEED_HAM)
SEED_REPEAT = 3   # give Indian examples a little extra weight

METRICS = None
if ds_texts:
    # Honest evaluation: test only on unseen dataset messages
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds_texts, ds_labels, test_size=0.2,
        random_state=42, stratify=ds_labels
    )
    eval_model = build_model()
    eval_model.fit(X_tr + seed_texts * SEED_REPEAT,
                   y_tr + seed_labels * SEED_REPEAT)
    pred = eval_model.predict(X_te)
    METRICS = {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision (spam)": precision_score(y_te, pred, pos_label=1),
        "Recall (spam)": recall_score(y_te, pred, pos_label=1),
        "F1-score (spam)": f1_score(y_te, pred, pos_label=1),
        "Test messages": len(y_te),
    }

# Final model uses all available data
model = build_model()
model.fit(ds_texts + seed_texts * SEED_REPEAT,
          ds_labels + seed_labels * SEED_REPEAT)

VEC = model.named_steps["tfidf"]
CLF = model.named_steps["clf"]
FEATURES = VEC.get_feature_names_out()
COEFS = CLF.coef_[0]

print(f"Model ready. Training source: {DATA_SOURCE}")


def ml_predict(text):
    """Return spam probability from the ML model."""
    return float(model.predict_proba([text])[0][1])


def top_spam_words(text, n=6):
    """Explainability: which words pushed the score towards spam."""
    x = VEC.transform([text]).tocoo()
    contrib = [
        (FEATURES[j], v * COEFS[j]) for j, v in zip(x.col, x.data)
    ]
    contrib = [c for c in contrib if c[1] > 0]
    contrib.sort(key=lambda c: c[1], reverse=True)
    return contrib[:n]


# ================================================================
# STAGE 2 — INDIAN SCAM RULE ENGINE
# ================================================================

RULES = [
    {
        "name": "Asks you to share OTP / PIN / CVV",
        "critical": True,
        "patterns": [
            r"\b(share|send|tell|forward|provide|give)\b.{0,20}\b(otp|one[\s-]?time password|pin|cvv|verification code)\b",
            r"\b(otp|cvv)\b.{0,25}\bto (complete|verify|receive|claim|redeem)\b",
        ],
        "why": "Banks, UPI apps and companies never ask you to share an OTP, PIN or CVV.",
    },
    {
        "name": "UPI PIN / QR to *receive* money",
        "critical": True,
        "patterns": [
            r"\b(enter|type|put)\b.{0,15}\b(upi\s*)?pin\b.{0,30}\b(receive|get|credit|refund)\b",
            r"\bcollect request\b",
            r"\bscan\b.{0,20}\bqr\b.{0,30}\b(receive|get)\b",
        ],
        "why": "A UPI PIN is only needed to SEND money. You never enter a PIN or scan a QR to receive money.",
    },
    {
        "name": "KYC / PAN / Aadhaar update or account block threat",
        "critical": False,
        "patterns": [
            r"\b(kyc|pan|aadhaar|aadhar)\b.{0,40}\b(update|expire|expired|link|linked|pending|verify)\b",
            r"\baccount\b.{0,30}\b(blocked|block|suspended|suspend|deactivated|locked|frozen|closed)\b",
        ],
        "why": "Scammers threaten account blocking to make you click a fake KYC link.",
    },
    {
        "name": "Electricity disconnection threat",
        "critical": False,
        "patterns": [
            r"\b(electricity|power|bijli)\b.{0,60}\b(disconnect|disconnected|cut|cut off|cutoff)\b",
            r"\bdisconnect(ed|ion)?\b.{0,40}\b(tonight|today|\d{1,2}[:.]\d{2})\b",
        ],
        "why": "Electricity boards do not send disconnection warnings asking you to call a personal number.",
    },
    {
        "name": "Lottery / prize / lucky draw",
        "critical": False,
        "patterns": [
            r"\b(won|winner|win)\b.{0,40}\b(rs\.?|₹|inr|lakh|crore|prize|lottery|reward|iphone|car)\b",
            r"\b(lottery|lucky draw|kbc|jackpot)\b",
        ],
        "why": "You cannot win a lottery or draw you never entered.",
    },
    {
        "name": "Fake courier / parcel problem",
        "critical": False,
        "patterns": [
            r"\b(parcel|package|courier|shipment)\b.{0,60}\b(not be delivered|undelivered|on hold|held|incomplete address|customs|update (your )?address)\b",
        ],
        "why": "Fake delivery messages lead to phishing pages or small 'redelivery fee' payments.",
    },
    {
        "name": "Digital arrest / police / CBI threat",
        "critical": True,
        "patterns": [
            r"\bdigital arrest\b",
            r"\b(cbi|police|narcotics|ncb|customs|cyber cell)\b.{0,60}\b(arrest|case|fir|warrant|drugs|illegal)\b",
        ],
        "why": "There is no 'digital arrest' in Indian law. Real agencies do not arrest or demand money over phone or video call.",
    },
    {
        "name": "SIM / mobile number block threat",
        "critical": False,
        "patterns": [
            r"\b(sim|mobile number)\b.{0,30}\b(blocked|deactivated|disconnected|suspended)\b",
            r"\btrai\b",
        ],
        "why": "TRAI does not call or message people to block their SIM.",
    },
    {
        "name": "Fake job / task-based earning",
        "critical": False,
        "patterns": [
            r"\b(work from home|part[\s-]?time job|daily income)\b",
            r"\bearn\b.{0,25}\b(daily|per day|a day|/day|per hour)\b",
            r"\blike\b.{0,15}\b(youtube|videos|instagram|posts)\b",
            r"\btelegram\b",
        ],
        "why": "Task-based job offers usually end with asking you to 'invest' or pay a deposit.",
    },
    {
        "name": "Instant / pre-approved loan offer",
        "critical": False,
        "patterns": [
            r"\b(pre[\s-]?approved|instant)\b.{0,20}\bloan\b",
            r"\bloan\b.{0,40}\b(no documents|0% interest|without cibil)\b",
        ],
        "why": "Unregulated loan apps often harass users and misuse their phone data.",
    },
    {
        "name": "Remote access app / APK install",
        "critical": True,
        "patterns": [
            r"\b(anydesk|teamviewer|quicksupport|rustdesk|airdroid)\b",
            r"\.apk\b",
            r"\b(download|install)\b.{0,25}\b(app|application)\b",
        ],
        "why": "Remote access apps and APK files let scammers control your phone and read your OTPs.",
    },
    {
        "name": "Reward points / free recharge bait",
        "critical": False,
        "patterns": [
            r"\breward points?\b.{0,40}\b(expire|expiring|redeem)\b",
            r"\b(free|cashback)\b.{0,30}\b(recharge|gift|voucher)\b",
        ],
        "why": "Fake reward and free recharge links are a common way to steal card details.",
    },
    {
        "name": "Urgency / pressure",
        "critical": False,
        "patterns": [
            r"\b(urgent|urgently|immediately|right now|asap|last chance|final notice|final warning|expires? today|before midnight|tonight)\b",
            r"\bwithin \d+\s*(hours?|hrs?|minutes?|mins?)\b",
        ],
        "why": "Scammers create panic so you act before thinking.",
    },
    {
        "name": "Secrecy / isolation",
        "critical": False,
        "patterns": [
            r"\b(do not|don'?t)\s+(tell|inform)\b.{0,20}\b(anyone|family|police|bank|anybody)\b",
            r"\bkeep (this|it) (secret|confidential)\b",
        ],
        "why": "Asking you to keep it secret stops you from checking with family or the bank.",
    },
    {
        "name": "Asks you to call / WhatsApp a personal number",
        "critical": False,
        "patterns": [
            r"\b(call|whatsapp|contact|message)\b.{0,30}(\+?91[\s-]?)?\b[6-9]\d{9}\b",
        ],
        "why": "Official organisations use published helplines, not personal mobile numbers.",
    },
]

NEGATION_RE = re.compile(
    r"\b(not|never|don'?t|do not|won'?t|no one|nobody)\b[^.!?]*$", re.I
)


def is_negated(text, start):
    """Skip matches like 'Do NOT share your OTP' (that is safe advice)."""
    window = text[max(0, start - 25):start]
    return bool(NEGATION_RE.search(window))


def check_rules(text):
    hits = []
    for rule in RULES:
        for pattern in rule["patterns"]:
            found = None
            for m in re.finditer(pattern, text, flags=re.I):
                if not is_negated(text, m.start()):
                    found = m.group(0)
                    break
            if found:
                hits.append({
                    "name": rule["name"],
                    "critical": rule["critical"],
                    "why": rule["why"],
                    "match": found.strip(),
                })
                break
    return hits


# ================================================================
# STAGE 3 — LINK ANALYSIS
# ================================================================

URL_RE = re.compile(
    r"((?:https?://|www\.)[^\s<>\"']+"
    r"|\b[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\."
    r"(?:com|in|net|org|xyz|top|click|link|live|online|site|info|ly|me|co|io|icu|tk|ml|ga|cf|gq|shop|buzz|app|sbi)"
    r"(?:/[^\s<>\"']*)?)",
    re.I,
)

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly",
    "rb.gy", "shorturl.at", "tiny.cc", "ow.ly", "rebrand.ly", "t.ly",
}

RISKY_TLDS = {
    "xyz", "top", "click", "link", "live", "online", "site", "icu",
    "tk", "ml", "ga", "cf", "gq", "buzz", "shop", "info",
}

# Official domains (prototype list — verify and extend for your report).
# Indian banks are also moving to the RBI-mandated ".bank.in" domain.
BRANDS = {
    "sbi": ["sbi.co.in", "onlinesbi.sbi", "sbi.bank.in"],
    "hdfc": ["hdfcbank.com", "hdfc.bank.in"],
    "icici": ["icicibank.com", "icici.bank.in"],
    "axis": ["axisbank.com", "axis.bank.in"],
    "paytm": ["paytm.com"],
    "amazon": ["amazon.in", "amazon.com"],
    "amzn": ["amazon.in", "amazon.com"],
    "flipkart": ["flipkart.com"],
    "indiapost": ["indiapost.gov.in"],
    "incometax": ["incometax.gov.in"],
    "epfo": ["epfindia.gov.in"],
    "jio": ["jio.com"],
    "airtel": ["airtel.in"],
}

ALL_OFFICIAL = {d for ds in BRANDS.values() for d in ds}


def domain_matches(host, domain):
    return host == domain or host.endswith("." + domain)


def analyze_urls(text):
    findings = []
    for raw in URL_RE.findall(text):
        url = raw.rstrip(".,);!?'\"")
        full = url if url.lower().startswith("http") else "http://" + url
        host = (urlparse(full).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if not host:
            continue

        official = (
            any(domain_matches(host, d) for d in ALL_OFFICIAL)
            or host.endswith(".bank.in")
            or host.endswith(".gov.in")
        )

        flags = []
        if url.lower().startswith("http://"):
            flags.append("Uses insecure `http://` (no padlock)")
        if host in SHORTENERS:
            flags.append("Link shortener hides the real destination")
        if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
            flags.append("Raw IP address instead of a website name")
        tld = host.rsplit(".", 1)[-1]
        if tld in RISKY_TLDS:
            flags.append(f"Risky domain ending `.{tld}` often used by scammers")
        if url.lower().split("?")[0].endswith(".apk"):
            flags.append("Downloads an Android app (.apk) — common malware route")
        if not official:
            for brand, domains in BRANDS.items():
                if brand in host:
                    flags.append(
                        f"Contains the name '{brand}' but is NOT an official "
                        f"{brand.upper()} website (lookalike domain)"
                    )
                    break
        if host.count("-") >= 2 or sum(c.isdigit() for c in host) >= 4:
            flags.append("Unusual domain with many hyphens or digits")

        findings.append({"url": url, "host": host, "flags": flags, "official": official})
    return findings


# ================================================================
# STAGE 4 — SENDER CHECK
# ================================================================

def analyze_sender(sender):
    """
    In India, banks, companies and government send bulk SMS from
    registered header IDs like 'AX-SBIINB' or 'VM-HDFCBK',
    not from personal 10-digit mobile numbers.
    """
    s = (sender or "").strip()
    if not s:
        return None

    compact = re.sub(r"[\s\-()]", "", s)
    if re.fullmatch(r"(\+?91|0)?[6-9]\d{9}", compact):
        return {
            "personal": True,
            "note": "Sent from a **personal 10-digit mobile number**. Banks, government "
                    "and companies send official SMS from header IDs (e.g. `AX-SBIINB`), "
                    "not personal numbers.",
        }
    if re.fullmatch(r"[A-Z]{2}-[A-Z0-9]{6}(-[A-Z])?", s.upper()):
        return {
            "personal": False,
            "note": "Sender looks like a **registered business header ID**. This is a good "
                    "sign, but still check the message content and links.",
        }
    return {"personal": False, "note": "Sender format not recognised."}


# ================================================================
# STAGE 5 — RAG: OFFICIAL GUIDANCE KNOWLEDGE BASE
# ================================================================

KNOWLEDGE_BASE = [
    {"title": "Banks never ask for OTP",
     "text": "Banks, RBI and UPI apps never ask customers to share OTP, PIN, CVV, password or card details by SMS, call or link. share otp pin cvv bank account reward redeem."},
    {"title": "UPI PIN is only for sending money",
     "text": "You only enter your UPI PIN to send or pay money. You never need a PIN or QR scan to receive money or a refund. upi pin receive refund collect request qr."},
    {"title": "KYC updates are done officially",
     "text": "KYC, PAN and Aadhaar updates are done through your bank branch, official bank app or official website, never through links in SMS. kyc pan aadhaar account blocked suspended update link."},
    {"title": "Electricity bill disconnection scams",
     "text": "Electricity boards do not ask you to call a personal mobile number to avoid disconnection tonight. Check your bill on the official board website or app. electricity power bill disconnected tonight officer."},
    {"title": "There is no 'digital arrest'",
     "text": "Police, CBI, customs or narcotics officers do not arrest people over video call or demand money to close a case. digital arrest cbi police drugs parcel case warrant."},
    {"title": "TRAI does not block SIMs by call or SMS",
     "text": "TRAI does not call or message users to disconnect their mobile number or SIM. sim blocked trai mobile number disconnected."},
    {"title": "Fake lottery and prize messages",
     "text": "Lotteries, KBC prizes or lucky draws you never entered are fake. Real prizes never ask for a fee or bank details. won lottery kbc prize lucky draw reward."},
    {"title": "Fake courier and delivery messages",
     "text": "Check parcel status only on the official courier or India Post website. Do not pay small redelivery fees through SMS links. parcel courier delivery address customs india post."},
    {"title": "Task-based job scams",
     "text": "Genuine employers do not pay you to like videos or ask for deposits. Telegram and WhatsApp task jobs are a common fraud. job work from home part time earn daily telegram youtube like."},
    {"title": "Loan app safety",
     "text": "Borrow only from RBI-regulated banks and NBFCs. Instant loan apps with no documents may misuse your contacts and photos. loan instant pre-approved no documents interest."},
    {"title": "Never install remote access apps for strangers",
     "text": "Apps like AnyDesk or TeamViewer and unknown APK files let fraudsters control your phone and read OTPs. anydesk teamviewer apk install app download."},
    {"title": "How to report cyber fraud in India",
     "text": "If you lost money, call the national cyber crime helpline 1930 immediately or report at cybercrime.gov.in. Suspected fraud messages can be reported on the Chakshu facility of the Sanchar Saathi portal. report fraud money lost helpline."},
]

kb_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
kb_matrix = kb_vectorizer.fit_transform(
    [k["title"] + " " + k["text"] for k in KNOWLEDGE_BASE]
)


def retrieve_guidance(query, top_k=3, min_score=0.05):
    q = kb_vectorizer.transform([query])
    scores = cosine_similarity(q, kb_matrix)[0]
    order = np.argsort(scores)[::-1][:top_k]
    results = [(KNOWLEDGE_BASE[i], float(scores[i])) for i in order if scores[i] >= min_score]
    # Always include reporting info if nothing else matched
    if not results:
        results = [(KNOWLEDGE_BASE[-1], 0.0)]
    return results


# ================================================================
# STAGE 6 — RISK SCORING AND REPORT
# ================================================================

def compute_risk(ml_prob, rule_hits, url_findings, sender_info):
    n_rules = len(rule_hits)
    n_critical = sum(1 for h in rule_hits if h["critical"])
    n_url_flags = sum(len(u["flags"]) for u in url_findings)

    sender_bad = bool(
        sender_info and sender_info["personal"] and (n_rules > 0 or url_findings)
    )

    evidence = min(
        1.0,
        0.15 * n_rules + 0.25 * n_critical + 0.15 * n_url_flags
        + (0.20 if sender_bad else 0.0)
    )

    score = 0.55 * ml_prob + 0.45 * evidence

    if n_critical >= 1:
        score = max(score, 0.75)
    if ml_prob >= 0.90:
        score = max(score, 0.70)

    # Only clean official links and no warning signs → lower the score
    if (url_findings and n_rules == 0
            and all(u["official"] and not u["flags"] for u in url_findings)):
        score = min(score, 0.35)

    score = float(min(score, 0.99))

    if score >= 0.70:
        level = "HIGH"
    elif score >= 0.40:
        level = "SUSPICIOUS"
    else:
        level = "LOW"

    return score, level, sender_bad


def action_plan(level):
    if level == "HIGH":
        steps = [
            "**Do not reply, call back or click any link** in this message.",
            "Never share OTP, UPI PIN, CVV, passwords or Aadhaar/PAN details.",
            "Do not install any app (AnyDesk, TeamViewer, APK files) the message asks for.",
            "Block the sender and report it on the **Chakshu** facility of the Sanchar Saathi portal.",
            "If you already paid or shared details: **call 1930 immediately**, inform your bank to block the card/UPI, and file a complaint at **cybercrime.gov.in**.",
        ]
    elif level == "SUSPICIOUS":
        steps = [
            "**Pause.** Do not click links or call numbers given in the message.",
            "Verify independently using the official app, website or helpline (from the back of your card or the official site).",
            "Never share OTP, PIN or CVV with anyone.",
            "If it turns out to be fake, block and report the sender.",
        ]
    else:
        steps = [
            "No strong warning signs found, but stay alert.",
            "Still never share OTP, PIN or CVV, even if the message looks genuine.",
            "If anything feels off, check through the official app or website.",
        ]
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))


def analyse(message, sender):
    message = (message or "").strip()
    if not message:
        return None, 0.0, "⚠️ Please paste an SMS message to analyse."

    # 1. ML model
    ml_prob = ml_predict(message)
    words = top_spam_words(message)

    # 2. Rules
    rule_hits = check_rules(message)

    # 3. Links
    url_findings = analyze_urls(message)

    # 4. Sender
    sender_info = analyze_sender(sender)

    # 5. Risk
    score, level, sender_bad = compute_risk(ml_prob, rule_hits, url_findings, sender_info)

    # 6. RAG guidance (query = message + detected categories)
    rag_query = message + " " + " ".join(h["name"] for h in rule_hits)
    guidance = retrieve_guidance(rag_query)

    # ---------- Build report ----------
    headline = {
        "HIGH": "🚨 **HIGH RISK — Likely Scam / Spam**",
        "SUSPICIOUS": "⚠️ **SUSPICIOUS — Verify Before Acting**",
        "LOW": "✅ **LOW RISK — No Strong Warning Signs**",
    }[level]

    safe_msg = html.escape(message).replace("\n", "<br>")

    if words:
        words_md = ", ".join(f"`{w}`" for w, _ in words)
    else:
        words_md = "No strongly spam-like words found."

    if rule_hits:
        rules_md = "\n".join(
            f"- {'🔴' if h['critical'] else '🟠'} **{h['name']}** — matched: "
            f"“{html.escape(h['match'])}”  \n  _{h['why']}_"
            for h in rule_hits
        )
    else:
        rules_md = "- No known Indian scam patterns matched. (This does not guarantee the message is safe.)"

    if url_findings:
        link_lines = []
        for u in url_findings:
            status = "✅ Official domain" if u["official"] and not u["flags"] else (
                "🔴 Suspicious" if u["flags"] else "⚪ Unknown domain")
            link_lines.append(f"- `{u['url']}` → **{status}** (domain: `{u['host']}`)")
            for f in u["flags"]:
                link_lines.append(f"  - {f}")
        links_md = "\n".join(link_lines)
    else:
        links_md = "- No links found in the message."

    if sender_info:
        sender_md = ("🔴 " if sender_bad else "") + sender_info["note"]
    else:
        sender_md = "Not provided."

    guidance_md = "\n".join(
        f"- **{g['title']}** — {g['text'].split('.')[0]}." +
        (f" _(relevance {s:.2f})_" if s > 0 else "")
        for g, s in guidance
    )

    report = f"""
## {headline}

**Risk score:** `{score:.2f} / 1.00`

> {safe_msg}

---

### 1. Machine Learning Model
- Spam probability: **{ml_prob:.2f}**
- Words that pushed the score towards spam: {words_md}

### 2. Scam Warning Signs
{rules_md}

### 3. Link Analysis
{links_md}

### 4. Sender Check
{sender_md}

### 5. Official Guidance (retrieved)
{guidance_md}

### 6. What You Should Do
{action_plan(level)}

---
_ScamShield AI is a student prototype for decision support. Its score is not a guarantee.
When in doubt, verify through official channels._
"""

    label = {"Scam / Spam": score, "Looks Safe": 1.0 - score}
    return label, round(score, 2), report


# ================================================================
# STAGE 7 — MODEL INFO TEXT
# ================================================================

if METRICS:
    metrics_md = "\n".join(
        f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |"
        for k, v in METRICS.items()
    )
    metrics_block = f"""
| Metric | Value |
|---|---|
{metrics_md}

_Evaluated on a held-out 20% test split of the dataset (not used for training)._
"""
else:
    metrics_block = (
        "⚠️ No dataset found, so the model is trained only on the small built-in "
        "sample set and cannot be evaluated. Download `spam.csv` (UCI SMS Spam "
        "Collection from Kaggle) into this folder and restart for a proper model."
    )

MODEL_INFO = f"""
## About the Model

**Training data:** {DATA_SOURCE}
+ {len(SEED_SPAM)} Indian scam and {len(SEED_HAM)} Indian normal example messages.

**Pipeline**
1. **TF-IDF vectoriser** (word unigrams + bigrams) converts text into numbers.
2. **Logistic Regression** (class-balanced) predicts spam probability.
3. **Rule engine** detects {len(RULES)} Indian scam patterns (with negation handling, so
   "Do NOT share your OTP" is not flagged).
4. **Link analyser** checks shorteners, lookalike brand domains, risky domain endings and APKs.
5. **Sender check** flags personal mobile numbers pretending to be organisations.
6. **RAG** retrieves official safety guidance using TF-IDF cosine similarity.
7. **Risk score** = 55% ML probability + 45% rule/link/sender evidence,
   with a minimum of 0.75 when a critical sign (OTP/PIN request, digital arrest,
   remote access app) is found.

### Evaluation
{metrics_block}

### Limitations
- English only; Hindi / Hinglish / Punjabi messages are less reliable.
- Rules and official domain list are a prototype and must be kept updated.
- Scammers constantly change wording (concept drift), so the model needs retraining.
"""


# ================================================================
# STAGE 8 — GRADIO INTERFACE
# ================================================================

EXAMPLES = [
    ["Dear customer, your SBI account will be blocked today. Update your KYC immediately: http://sbi-kyc-update.xyz", "+91 98765 43210"],
    ["Your electricity connection will be disconnected tonight at 9:30 PM. Call our officer 9876543210 immediately.", "9876543210"],
    ["You have received Rs 4,999 refund. Enter your UPI PIN to receive the money.", ""],
    ["Your OTP for login is 482913. Do not share it with anyone. -SBI", "AX-SBIINB"],
    ["Hey, class is cancelled tomorrow. See you on Monday!", "9814012345"],
    ["Congratulations! You won an iPhone 16 in our lucky draw. Claim now: bit.ly/win-iph16", ""],
]

with gr.Blocks(title="ScamShield AI — SMS", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🛡️ ScamShield AI — SMS Scam & Spam Detector
        Paste an SMS to check it using machine learning, Indian scam pattern detection,
        link analysis and official safety guidance. **No API keys, runs fully offline.**
        """
    )

    with gr.Tab("Check an SMS"):
        with gr.Row():
            with gr.Column(scale=1):
                msg_in = gr.Textbox(
                    label="SMS message",
                    lines=6,
                    placeholder="Paste the SMS here...",
                )
                sender_in = gr.Textbox(
                    label="Sender (optional)",
                    placeholder="e.g. AX-SBIINB or +91 98765 43210",
                )
                with gr.Row():
                    analyse_btn = gr.Button("Analyse", variant="primary")
                    clear_btn = gr.ClearButton(value="Clear")
                gr.Examples(examples=EXAMPLES, inputs=[msg_in, sender_in])

            with gr.Column(scale=2):
                verdict_out = gr.Label(label="Verdict", num_top_classes=2)
                score_out = gr.Number(label="Risk score (0–1)", precision=2, interactive=False)
                report_out = gr.Markdown()

        clear_btn.add([msg_in, sender_in, verdict_out, score_out, report_out])

        analyse_btn.click(
            fn=analyse,
            inputs=[msg_in, sender_in],
            outputs=[verdict_out, score_out, report_out],
        )

    with gr.Tab("About the model"):
        gr.Markdown(MODEL_INFO)

    gr.Markdown(
        "_Lost money to a scam? Call **1930** or report at **cybercrime.gov.in**._"
    )


if __name__ == "__main__":
    demo.launch()