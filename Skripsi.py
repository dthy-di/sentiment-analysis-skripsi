# ================= IMPORT =================
import os
import re
import json
import threading
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
from afinn import Afinn

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# ================= REGEX =================
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9\s]")


# ================= PREPROCESS =================
def preprocess_en(text: str) -> str:
    t = str(text).lower()
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


# ================= TRANSLATE =================
def make_translator():
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target="en")
    except Exception:
        return None


def translate_text_safe(translator, text: str):
    if translator is None:
        return text
    try:
        return translator.translate(text)
    except Exception:
        return text


# ================= LOAD FILE =================
def load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    raise ValueError("Format file tidak didukung")


# ================= AFINN =================
def afinn_label(text: str, afinn: Afinn):
    score = afinn.score(text)

    if score > 0:
        return score, "positive"
    elif score < 0:
        return score, "negative"
    else:
        return score, "neutral"


# ================= BALANCE DATA =================
def balance_binary(df):
    df_bin = df[df["label"].isin(["positive", "negative"])]

    pos = df_bin[df_bin["label"] == "positive"]
    neg = df_bin[df_bin["label"] == "negative"]

    if len(pos) == 0 or len(neg) == 0:
        return df_bin

    n = min(len(pos), len(neg))

    df_bal = pd.concat([
        pos.sample(n, random_state=42),
        neg.sample(n, random_state=42)
    ])

    return df_bal.sample(frac=1, random_state=42)

# ================= MODEL =================
def train_nb(df: pd.DataFrame) -> Pipeline:

    # 🔥 Ambil semua label (positive, negative, neutral)
    df_multi = df[df["label"].isin(["positive", "negative", "neutral"])]

    if len(df_multi) < 3:
        raise ValueError("Data tidak cukup untuk training 3 kelas")

    X = df_multi["text_processed_en"]
    y = df_multi["label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2
        )),
        ("nb", MultinomialNB(alpha=0.5))
    ])

    model.fit(X, y)
    return model





# ================= GUI =================
class SentimentGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Analysis ")
        self.geometry("1300x720")

        self.df_loaded: Optional[pd.DataFrame] = None
        self.df_afinn: Optional[pd.DataFrame] = None
        self.model: Optional[Pipeline] = None

        self._build_ui()

    # ================= SAVE =================
    def save_result(self):
        if self.df_afinn is None:
            messagebox.showwarning("Warning", "Belum ada data")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv")]
        )
        if not path:
            return

        self.df_afinn.to_csv(path, index=False)
        messagebox.showinfo("Success", "Data berhasil disimpan!")

    # ================= UI =================
    def _build_ui(self):
        pad = {"padx": 8, "pady": 5}

        frm_in = ttk.LabelFrame(self, text="Input Data")
        frm_in.pack(fill="x", **pad)

        ttk.Button(frm_in, text="Pilih File", command=self.on_browse)\
            .grid(row=0, column=0)

        self.lbl_file = ttk.Label(frm_in, text="Belum ada file")
        self.lbl_file.grid(row=0, column=1, sticky="w")

        ttk.Label(frm_in, text="Kolom komentar").grid(row=1, column=0)
        self.combo_col = ttk.Combobox(frm_in, state="readonly", width=40)
        self.combo_col.grid(row=1, column=1, sticky="w")

        self.var_translate = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_in, text="Translate ID → EN", variable=self.var_translate)\
            .grid(row=2, column=0, columnspan=2, sticky="w")

        ttk.Label(frm_in, text="Preview Data").grid(row=3, column=0)
        self.var_preview = tk.IntVar(value=30)
        ttk.Entry(frm_in, textvariable=self.var_preview, width=10)\
            .grid(row=3, column=1, sticky="w")

        self.progress = ttk.Progressbar(frm_in, mode="indeterminate")
        self.progress.grid(row=4, column=0, columnspan=2, sticky="ew")

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)

        # ================= AFINN =================
        frm_afinn = ttk.LabelFrame(main, text="1️⃣ Labeling AFINN")
        frm_afinn.pack(side="left", fill="both", expand=True, padx=10)

        ttk.Button(frm_afinn, text="Proses AFINN", command=self.run_afinn)\
            .pack(anchor="w", padx=10, pady=5)

        self.tree_afinn = ttk.Treeview(
            frm_afinn,
            columns=("text", "label"),
            show="headings"
        )
        self.tree_afinn.heading("text", text="text_processed_en")
        self.tree_afinn.heading("label", text="label_afinn")
        self.tree_afinn.pack(fill="both", expand=True, padx=10, pady=5)

        self.lbl_dist = ttk.Label(frm_afinn, text="Distribusi NB: -")
        self.lbl_dist.pack(anchor="w", padx=10)

        # ================= NB =================
        frm_nb = ttk.LabelFrame(main, text="2️⃣ Naive Bayes")
        frm_nb.pack(side="right", fill="both", expand=True, padx=10)

        ttk.Button(frm_nb, text="Train & Prediksi NB", command=self.run_nb)\
            .pack(anchor="w", padx=10, pady=5)

        ttk.Button(frm_nb, text="Simpan Hasil", command=self.save_result)\
            .pack(anchor="w", padx=10, pady=5)

        self.tree_nb = ttk.Treeview(
            frm_nb,
            columns=("text", "label", "pred"),
            show="headings"
        )
        self.tree_nb.heading("text", text="text_processed_en")
        self.tree_nb.heading("label", text="label_afinn")
        self.tree_nb.heading("pred", text="prediksi_NB")
        self.tree_nb.pack(fill="both", expand=True, padx=10, pady=5)

        ttk.Label(frm_nb, text="Komentar baru").pack(anchor="w", padx=10)

        self.txt_new = tk.Text(frm_nb, height=4)
        self.txt_new.pack(fill="x", padx=10)

        ttk.Button(frm_nb, text="Prediksi", command=self.predict_new)\
            .pack(anchor="w", padx=10, pady=5)

        self.var_result = tk.StringVar(value="-")
        ttk.Label(frm_nb, textvariable=self.var_result,
                  font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10)

    # ================= FILE =================
    def on_browse(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        self.df_loaded = load_dataframe(path)
        self.lbl_file.config(text=os.path.basename(path))
        self.combo_col["values"] = list(self.df_loaded.columns)
        self.combo_col.set(self.df_loaded.columns[0])

    # ================= AFINN =================
    def run_afinn(self):
        threading.Thread(target=self._run_afinn_worker, daemon=True).start()

    def _run_afinn_worker(self):
        if self.df_loaded is None:
            return

        self.progress.start()
        self.tree_afinn.delete(*self.tree_afinn.get_children())

        col = self.combo_col.get()
        afinn = Afinn()
        translator = make_translator() if self.var_translate.get() else None

        df = self.df_loaded.copy()
        df["text_en"] = df[col].astype(str).apply(
            lambda x: translate_text_safe(translator, x)
        )
        df["text_processed_en"] = df["text_en"].apply(preprocess_en)

        scored = df["text_processed_en"].apply(lambda t: afinn_label(t, afinn))
        df["label"] = scored.apply(lambda x: x[1])

        self.df_afinn = df

        # 🔥 Hitung distribusi binary NB
        df_bin = df[df["label"].isin(["positive", "negative"])]
        total = len(df_bin)
        # 🔥 Hitung distribusi 3 kelas
        total = len(df)

        if total > 0:
            pos = len(df[df["label"] == "positive"])
            neg = len(df[df["label"] == "negative"])
            neu = len(df[df["label"] == "neutral"])

            self.lbl_dist.config(
                text=f"Distribusi → "
                     f"Positive: {(pos / total) * 100:.2f}% | "
                     f"Negative: {(neg / total) * 100:.2f}% | "
                     f"Neutral: {(neu / total) * 100:.2f}%"
            )

        self.after(0, self._update_afinn_view)

    def _update_afinn_view(self):
        limit = self.var_preview.get()
        for _, r in self.df_afinn.head(limit).iterrows():
            self.tree_afinn.insert(
                "", "end",
                values=(r["text_processed_en"][:400], r["label"])
            )
        self.progress.stop()

    # ================= NB =================
    def run_nb(self):
        threading.Thread(target=self._run_nb_worker, daemon=True).start()

    def _run_nb_worker(self):
        if self.df_afinn is None:
            return

        try:
            self.progress.start()

            self.model = train_nb(self.df_afinn)

            # 🔥 CEK MODEL ADA DULU
            if self.model is None:
                raise ValueError("Model gagal dibuat")

            print("Classes:", self.model.named_steps["nb"].classes_)
            print("Class log prior:", self.model.named_steps["nb"].class_log_prior_)

            self.df_afinn["prediksi"] = self.model.predict(
                self.df_afinn["text_processed_en"]
            )

            self.after(0, self._update_nb_view)

        except Exception as e:
            self.after(0, lambda: messagebox.showwarning("NB Error", str(e)))

    def _update_nb_view(self):
        self.tree_nb.delete(*self.tree_nb.get_children())
        limit = self.var_preview.get()
        for _, r in self.df_afinn.head(limit).iterrows():
            self.tree_nb.insert(
                "", "end",
                values=(r["text_processed_en"][:400],
                        r["label"],
                        r.get("prediksi", "-"))
            )
        self.progress.stop()

    # ================= PREDICT =================
    def predict_new(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Train NB dulu")
            return

        text = self.txt_new.get("1.0", "end").strip()
        if not text:
            return

        text_proc = preprocess_en(text)
        pred = self.model.predict([text_proc])[0]
        self.var_result.set(pred)


if __name__ == "__main__":
    SentimentGUI().mainloop()

