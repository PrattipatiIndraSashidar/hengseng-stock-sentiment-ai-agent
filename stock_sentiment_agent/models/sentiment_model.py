"""Multilingual NLP Sentiment Analysis supporting English, Traditional Chinese, and Korean.

Three-tier architecture:
1. Tier 1 (Transformer Runtime): Hugging Face `cardiffnlp/twitter-xlm-roberta-base-sentiment`
   when `transformers` and `torch` are installed.
2. Tier 2 (Statistical Multilingual NLP Model — Active Default without PyTorch):
   `MultilingualFinancialNLPModel` (`multilingual-tfidf-logreg-hybrid-v2`), an L2-regularized
   Multinomial Logistic Regression (Softmax) classifier trained over L2-normalized TF-IDF
   features (Unicode script tokens + CJK/Hangul 2–3 character n-grams + negation-scoped
   tokens) combined with contrastive-clause weighting and curated financial lexicon priors.
3. Tier 3 (Documented Lexicon Fallback): Deterministic `multilingual-financial-lexicon-v1`
   when `use_nlp=False` or statistical inference fails.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from stock_sentiment_agent.preprocessing.data_cleaning import clean_text

EN_POS = {
    "beat", "beats", "bullish", "surge", "surged", "rally", "rallied", "growth",
    "profit", "profitable", "upgrade", "upgraded", "outperform", "strong", "gain",
    "gains", "record", "breakout", "expand", "expansion", "buy", "long", "optimistic",
    "rebound", "recovery", "dividend", "accelerat", "upside", "buyback", "resilient",
    "overweight", "momentum", "exceeded",
}
EN_NEG = {
    "miss", "missed", "bearish", "drop", "dropped", "plunge", "plunged", "slump",
    "loss", "losses", "downgrade", "downgraded", "underperform", "weak", "decline",
    "declines", "lawsuit", "fine", "regulatory", "crackdown", "sell", "short",
    "risk", "warning", "cut", "cuts", "layoff", "debt", "downside", "headwind",
    "selloff", "pressure", "underweight", "probe", "antitrust",
}
ZH_POS = [
    "利好", "大漲", "看漲", "增長", "盈利", "超預期", "勝預期", "突破", "買入", "做多", "加倉",
    "反彈", "強勁", "回購", "目標價上調", "上調目標價", "牛市", "創高", "業績亮眼", "受惠", "領漲",
    "大升", "升穿", "派息", "吸納", "買盤", "好倉", "看高", "重磅", "理想", "回勇",
]
ZH_NEG = [
    "利空", "利淡", "大跌", "看空", "看淡", "下跌", "虧損", "不及預期", "遜預期", "減倉", "賣出",
    "做空", "沽空", "監管", "罰款", "風險", "暴跌", "疲弱", "下調", "熊市", "承壓", "裁員",
    "抛售", "沽壓", "跌穿", "蟹貨", "止蝕", "淡倉", "疲軟", "萎縮", "減持",
]
KO_POS = [
    "상승", "급등", "호재", "실적 개선", "어닝 서프라이즈", "서프라이즈", "매수", "순매수",
    "강세", "돌파", "반등", "성장", "흑자", "목표가 상향", "상향", "수주", "신고가", "확대",
    "긍정", "수혜", "저평가", "밸류업", "배당", "자사주", "매집", "상한가", "기대감",
]
KO_NEG = [
    "하락", "급락", "악재", "실적 부진", "어닝 쇼크", "쇼크", "매도", "순매도", "약세",
    "이탈", "공매도", "적자", "하향", "리스크", "규제", "우려", "둔화", "손절", "부담",
    "고평가", "하한가", "투매", "과매도", "감소", "폭락", "경고",
]
NEGATIONS = {
    "not", "no", "never", "neither", "without", "hardly", "unlikely", "fail", "failed",
    "不", "沒", "沒有", "非", "未", "無", "並非", "別",
    "안", "못", "아니", "아닌", "없", "불가", "부족",
}
CONTRAST_MARKERS = ("however", "although", "but", "yet", "但", "不過", "然而", "惟", "하지만", "그러나", "반면")

# Balanced multilingual financial corpus (72 sentences: 24 EN + 24 ZH-HK + 24 KO; 8 Pos / 8 Neg / 8 Neu each)
MULTILINGUAL_TRAINING_CORPUS: List[Tuple[str, str, str]] = [
    # English Positive (8)
    ("en", "Positive", "Strong quarterly earnings beat estimates with cloud revenue surging and buyback expanded."),
    ("en", "Positive", "Analysts upgraded the stock to outperform with higher target price on AI margin expansion."),
    ("en", "Positive", "Shares rallied on record profit growth, resilient cash flow, and higher dividend payout."),
    ("en", "Positive", "Institutional inflows accelerate as breakout above resistance confirms bullish momentum."),
    ("en", "Positive", "Operating margin recovery and strong user monetization beat consensus forecasts."),
    ("en", "Positive", "Management raised full year guidance supported by solid enterprise demand."),
    ("en", "Positive", "Not weak at all, earnings surged ahead of forecasts with healthy upside."),
    ("en", "Positive", "Dividend hike and aggressive share buyback program boosted investor optimism."),
    # English Negative (8)
    ("en", "Negative", "Revenue missed consensus and management cut margin guidance amid fierce competition."),
    ("en", "Negative", "Shares plunged after regulatory probe, antitrust fine, and analyst downgrade to underperform."),
    ("en", "Negative", "Bearish pressure persists as slowing consumer demand triggers heavy institutional selloff."),
    ("en", "Negative", "Widening net loss, debt burden, and layoff warnings weigh on near-term valuation."),
    ("en", "Negative", "No rebound expected yet as headwinds and rising short interest push price below support."),
    ("en", "Negative", "Disappointing guidance and weak ad growth led to steep downside risk."),
    ("en", "Negative", "Initial gains faded; however regulatory crackdown and margin compression hurt sentiment."),
    ("en", "Negative", "Credit downgrade and steep revenue decline fueled heavy short selling pressure."),
    # English Neutral (8)
    ("en", "Neutral", "Company announced annual general meeting schedule and routine board committee changes."),
    ("en", "Neutral", "Trading volume remained average ahead of next week quarterly financial results release."),
    ("en", "Neutral", "Stock closed flat in range-bound session as investors await macroeconomic data."),
    ("en", "Neutral", "Filing confirmed unchanged share capital following routine employee option exercise."),
    ("en", "Neutral", "Exchange issued standard notice regarding adjusted reference price for technical event."),
    ("en", "Neutral", "Board scheduled regular quarterly briefing for institutional analysts next Thursday."),
    ("en", "Neutral", "Shares hovered near opening levels with balanced buy and sell order flow."),
    ("en", "Neutral", "Regulatory disclosure listed routine director shareholdings without material changes."),
    # Traditional Chinese / HKEX Positive (8)
    ("zh", "Positive", "騰訊業績勝預期，遊戲及廣告收入增長強勁，大行上調目標價看漲。"),
    ("zh", "Positive", "阿里巴巴宣布擴大回購兼派息，大戶資金買入推動股價大升突破阻力位。"),
    ("zh", "Positive", "小米電動車交付創高，盈利表現亮眼，市場看好估值重估空間，積極做多。"),
    ("zh", "Positive", "友邦保險新業務價值增長超預期，股價反彈領漲藍籌，大行維持買入評級。"),
    ("zh", "Positive", "美團受惠外賣及到店利潤改善，業績勝預期，北水持續淨買入加倉。"),
    ("zh", "Positive", "港股低開高走，藍籌獲買盤吸納，成交配合升穿關鍵阻力，後市偏強。"),
    ("zh", "Positive", "利好政策出台帶動科技股回勇，資金做多意欲濃厚，短線反彈浪確立。"),
    ("zh", "Positive", "大行上調目標價並維持買入評級，盈利前景理想帶動股價強勢突破。"),
    # Traditional Chinese / HKEX Negative (8)
    ("zh", "Negative", "業績不及預期兼毛利率受壓，大行下調目標價，市場沽壓沉重。"),
    ("zh", "Negative", "遭監管機構罰款及反壟斷調查，投資者恐慌抛售，股價大跌失守支持位。"),
    ("zh", "Negative", "虧損擴大及裁員消息拖累氣氛，淡友大舉沽空，短線走勢疲弱看淡。"),
    ("zh", "Negative", "阿里巴巴沽壓未散，高估值科技股遭減倉，蟹貨重重需嚴守止蝕。"),
    ("zh", "Negative", "訂單增速放緩且面臨激烈價格戰，盈利前景遜預期，股價暴跌穿底。"),
    ("zh", "Negative", "外資持續淨賣出減倉，市場氣氛轉熊，未見明顯買盤承接。"),
    ("zh", "Negative", "雖然早段曾反彈，但尾市沽盤湧現，業績指引下調令股價承壓。"),
    ("zh", "Negative", "受監管風險及利空消息影響，淡倉大增並引致大跌破位。"),
    # Traditional Chinese / HKEX Neutral (8)
    ("zh", "Neutral", "公司公布董事會會議日期，將於下月中旬審議季度業績及股息安排。"),
    ("zh", "Neutral", "港股全日窄幅上落，成交淡靜，投資者觀望聯儲局議息結果。"),
    ("zh", "Neutral", "聯交所發布例行通告，調整相關衍生產品及股份期權參考價。"),
    ("zh", "Neutral", "股份全日在開市價附近橫行整固，好淡雙方暫未分出勝負。"),
    ("zh", "Neutral", "公司刊發例行翌日披露報表，股本結構與上個交易日一致。"),
    ("zh", "Neutral", "股東周年大會通函已按上市規則寄發予登記股東閱覽。"),
    ("zh", "Neutral", "主板成交額與過去五日平均相若，市場靜待下周業績發布。"),
    ("zh", "Neutral", "公司確認首席財務官委任生效日期及常規委員會名單更新。"),
    # Korean / KRX Positive (8)
    ("ko", "Positive", "삼성전자 외국인 순매수와 함께 HBM 수주 확대 기대감으로 주가 급등 돌파."),
    ("ko", "Positive", "SK하이닉스 어닝 서프라이즈 기록, 증권사 목표가 상향 조정하며 신고가 경신."),
    ("ko", "Positive", "현대차 배당 확대 및 밸류업 프로그램 호재로 기관 매수세 유입 강세."),
    ("ko", "Positive", "네이버 AI 검색 수익화 및 실적 개선 확인, 저평가 매력 부각되며 반등."),
    ("ko", "Positive", "카카오 영업이익 흑자 전환 및 자사주 매입 발표로 투자심리 긍정적 전환."),
    ("ko", "Positive", "대규모 공급 계약 수주 소식에 매집세 유입되며 강한 상승 탄력 유지."),
    ("ko", "Positive", "실적 부진 우려가 있었으나 예상보다 빠른 수요 회복으로 목표가 상향 호재."),
    ("ko", "Positive", "자사주 소각과 배당 확대 호재로 외국인 매수세 유입되며 강세 마감."),
    # Korean / KRX Negative (8)
    ("ko", "Negative", "실적 부진과 어닝 쇼크로 외국인 순매도 급증, 주가 지지선 이탈 하락."),
    ("ko", "Negative", "공매도 물량 급증 및 고평가 논란 확산으로 개인 투매 이어지며 급락."),
    ("ko", "Negative", "규제 리스크 부각과 수익성 둔화 우려에 목표가 하향 리포트 속출 약세."),
    ("ko", "Negative", "영업적자 확대 및 성장 둔화 경고로 손절 물량 출회되며 신저가 위협."),
    ("ko", "Negative", "악재 겹치며 투자심리 냉각, 외국인 매도 공세에 반등 없이 하락 지속."),
    ("ko", "Negative", "업황 회복 지연 및 마진 압박 부담으로 단기 약세 흐름 불가피."),
    ("ko", "Negative", "장 초반 반등 기대했으나 외국인 매도세와 악재 겹쳐 약세 마감."),
    ("ko", "Negative", "적자 지속과 공매도 증가 부담으로 투매 출회되며 급락세 지속."),
    # Korean / KRX Neutral (8)
    ("ko", "Neutral", "정기 주주총회 소집 공고 및 사외이사 선임 안건 이사회 의결 발표."),
    ("ko", "Neutral", "금일 거래량 평이하며 주요 거시경제 지표 발표 앞두고 보합권 횡보 마감."),
    ("ko", "Neutral", "단기 급등 이후 숨고르기 장세 진입, 매수 매도 공방 속 관망세 유지."),
    ("ko", "Neutral", "분기 실적 발표 일정 공시 및 특이 사항 없는 보합 흐름 지속."),
    ("ko", "Neutral", "거래소 정규시장 기준가 및 배당락 관련 일반 안내 공지."),
    ("ko", "Neutral", "이사회 정기 일정 및 사내 규정 개정 관련 단순 정정 공시."),
    ("ko", "Neutral", "금일 장중 큰 변동성 없이 시가 부근에서 횡보하며 거래 마감."),
    ("ko", "Neutral", "주주명부 폐쇄 기준일 설정 및 정기 공시 일정 안내 공고."),
]


@dataclass
class SentimentPrediction:
    label: str  # Positive | Negative | Neutral
    score: float  # [-1.0, +1.0]
    positive_prob: float
    negative_prob: float
    neutral_prob: float
    confidence: float
    engine: str


class MultilingualFinancialNLPModel:
    """L2-normalized TF-IDF + Regularized Multinomial Logistic Regression (EN/ZH-HK/KO)."""

    CLASSES = ("Positive", "Negative", "Neutral")
    ENGINE_ID = "multilingual-tfidf-logreg-hybrid-v2"

    def __init__(self, l2_reg: float = 0.015, lr: float = 1.2, epochs: int = 55) -> None:
        self.l2_reg = l2_reg
        self.lr = lr
        self.epochs = epochs
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []
        self.weights: List[List[float]] = []  # shape [3][n_feats]
        self.biases: List[float] = [0.0, 0.0, 0.0]
        self.fit(MULTILINGUAL_TRAINING_CORPUS)

    @staticmethod
    def extract_features(text: str) -> List[str]:
        """Extract script tokens, negation-scoped tokens, CJK/Hangul n-grams, and domain cues."""
        lower = clean_text(text).lower()
        if not lower:
            return []

        raw_tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+|[\uac00-\ud7af]+", lower)
        features: List[str] = []

        for idx, tok in enumerate(raw_tokens):
            negated = (idx > 0 and raw_tokens[idx - 1] in NEGATIONS) or (
                idx > 1 and raw_tokens[idx - 2] in NEGATIONS
            )
            prefix = "NEG_" if negated else ""

            if re.match(r"^[a-z0-9_]+$", tok):
                features.append(f"{prefix}w:{tok}")
                if len(tok) >= 4:
                    features.append(f"{prefix}stem:{tok[:4]}")
            else:
                features.append(f"{prefix}w:{tok}")
                chars = list(tok)
                for n in (2, 3):
                    if len(chars) >= n:
                        for i in range(len(chars) - n + 1):
                            features.append(f"{prefix}g{n}:{''.join(chars[i : i + n])}")
                if len(chars) == 1:
                    features.append(f"{prefix}g1:{tok}")

        if re.search(r"\+\s*\d+(\.\d+)?%", lower):
            features.append("num:pct_up")
        if re.search(r"-\s*\d+(\.\d+)?%", lower):
            features.append("num:pct_down")

        return features

    def _vectorize(self, feats: Sequence[str]) -> Dict[int, float]:
        counts: Counter = Counter()
        for f in feats:
            idx = self.vocab.get(f)
            if idx is not None:
                counts[idx] += 1
        if not counts:
            return {}
        vec = {idx: (1.0 + math.log(cnt)) * self.idf[idx] for idx, cnt in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {idx: v / norm for idx, v in vec.items()}

    def fit(self, samples: Sequence[Tuple[str, str, str]]) -> None:
        docs_feats: List[List[str]] = []
        labels: List[int] = []
        df: Counter = Counter()

        for _lang, lbl, text in samples:
            if lbl not in self.CLASSES:
                continue
            feats = self.extract_features(text)
            docs_feats.append(feats)
            labels.append(self.CLASSES.index(lbl))
            for f in set(feats):
                df[f] += 1

        n_docs = max(1, len(docs_feats))
        vocab_terms = sorted(df.keys())
        self.vocab = {term: idx for idx, term in enumerate(vocab_terms)}
        self.idf = [math.log((n_docs + 1.0) / (df[term] + 1.0)) + 1.0 for term in vocab_terms]
        n_feats = len(vocab_terms)
        n_classes = len(self.CLASSES)

        X_vecs = [self._vectorize(feats) for feats in docs_feats]
        self.weights = [[0.0] * n_feats for _ in range(n_classes)]
        self.biases = [0.0] * n_classes
        decay = 1.0 - self.lr * self.l2_reg
        inv_n = self.lr / n_docs

        for _epoch in range(self.epochs):
            grad_b = [0.0] * n_classes
            sparse_grads: List[Dict[int, float]] = [{} for _ in range(n_classes)]

            for vec, y_idx in zip(X_vecs, labels):
                logits = [self.biases[c] + sum(self.weights[c][j] * val for j, val in vec.items()) for c in range(n_classes)]
                max_l = max(logits)
                exps = [math.exp(l - max_l) for l in logits]
                z = sum(exps) or 1.0
                probs = [e / z for e in exps]

                for c in range(n_classes):
                    err = probs[c] - (1.0 if c == y_idx else 0.0)
                    grad_b[c] += err
                    sg = sparse_grads[c]
                    for j, val in vec.items():
                        sg[j] = sg.get(j, 0.0) + err * val

            for c in range(n_classes):
                self.biases[c] -= inv_n * grad_b[c]
                wc = self.weights[c]
                for j in range(n_feats):
                    wc[j] *= decay
                for j, g in sparse_grads[c].items():
                    wc[j] -= inv_n * g

    def predict_proba(self, text: str) -> Dict[str, float]:
        feats = self.extract_features(text)
        vec = self._vectorize(feats)
        if not vec:
            return {"Positive": 0.25, "Negative": 0.25, "Neutral": 0.50}

        logits = [
            self.biases[c] + sum(self.weights[c][j] * val for j, val in vec.items())
            for c in range(len(self.CLASSES))
        ]
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        z = sum(exps) or 1.0
        return {cls: exps[idx] / z for idx, cls in enumerate(self.CLASSES)}

    @classmethod
    def run_cross_validation(cls) -> Dict[str, Any]:
        """Leave-one-out out-of-sample evaluation across EN, ZH-HK, and KO corpus."""
        n = len(MULTILINGUAL_TRAINING_CORPUS)
        correct = 0
        per_lang: Dict[str, List[bool]] = {"en": [], "zh": [], "ko": []}
        hybrid_correct = 0
        hybrid_per_lang: Dict[str, List[bool]] = {"en": [], "zh": [], "ko": []}

        for hold_idx in range(n):
            lang, true_lbl, text = MULTILINGUAL_TRAINING_CORPUS[hold_idx]
            train_slice = [row for i, row in enumerate(MULTILINGUAL_TRAINING_CORPUS) if i != hold_idx]
            model = cls()
            model.fit(train_slice)
            probs = model.predict_proba(text)
            pred_lbl = max(probs, key=lambda k: probs[k])
            hit = pred_lbl == true_lbl
            correct += int(hit)
            per_lang.setdefault(lang, []).append(hit)

            # Evaluate full hybrid analyzer (trained strictly on n-1 slice)
            analyzer = MultilingualSentimentAnalyzer(use_hf=False, use_nlp=False)
            analyzer._nlp_model = model
            analyzer.engine_name = cls.ENGINE_ID
            h_pred = analyzer.analyze_text(text)
            h_hit = h_pred.label == true_lbl
            hybrid_correct += int(h_hit)
            hybrid_per_lang.setdefault(lang, []).append(h_hit)

        return {
            "engine": cls.ENGINE_ID,
            "total_samples": n,
            "pure_nlp_loo_accuracy_pct": round((correct / max(1, n)) * 100.0, 2),
            "hybrid_nlp_loo_accuracy_pct": round((hybrid_correct / max(1, n)) * 100.0, 2),
            "per_language_pure_nlp_pct": {
                lang: round((sum(vals) / max(1, len(vals))) * 100.0, 2)
                for lang, vals in per_lang.items()
            },
            "per_language_hybrid_pct": {
                lang: round((sum(vals) / max(1, len(vals))) * 100.0, 2)
                for lang, vals in hybrid_per_lang.items()
            },
        }


_DEFAULT_NLP_INSTANCE: Optional[MultilingualFinancialNLPModel] = None


class MultilingualSentimentAnalyzer:
    """Multilingual sentiment scorer with HF Transformer, Statistical NLP, and Lexicon fallback."""

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        use_hf: bool = True,
        use_nlp: bool = True,
    ) -> None:
        global _DEFAULT_NLP_INSTANCE
        self.model_name = model_name
        self._hf_pipeline = None
        self._nlp_model: Optional[MultilingualFinancialNLPModel] = None
        self.engine_name = "multilingual-financial-lexicon-v1"

        if use_hf:
            try:
                from transformers import pipeline  # type: ignore

                self._hf_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name,
                    top_k=None,
                    truncation=True,
                    max_length=256,
                )
                self.engine_name = model_name
            except Exception:
                self._hf_pipeline = None

        if self._hf_pipeline is None and use_nlp:
            try:
                if _DEFAULT_NLP_INSTANCE is None:
                    _DEFAULT_NLP_INSTANCE = MultilingualFinancialNLPModel()
                self._nlp_model = _DEFAULT_NLP_INSTANCE
                self.engine_name = MultilingualFinancialNLPModel.ENGINE_ID
            except Exception:
                self._nlp_model = None

    def analyze_text(self, text: str) -> SentimentPrediction:
        cleaned = clean_text(text)
        if not cleaned:
            return SentimentPrediction("Neutral", 0.0, 0.2, 0.2, 0.6, 0.3, self.engine_name)

        if self._hf_pipeline is not None:
            try:
                raw_scores = self._hf_pipeline(cleaned)[0]
                probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                for entry in raw_scores:
                    lbl = str(entry["label"]).lower()
                    if "pos" in lbl or lbl == "label_2":
                        probs["positive"] = float(entry["score"])
                    elif "neg" in lbl or lbl == "label_0":
                        probs["negative"] = float(entry["score"])
                    else:
                        probs["neutral"] = float(entry["score"])
                net_score = max(-1.0, min(1.0, probs["positive"] - probs["negative"]))
                if probs["positive"] >= probs["negative"] and probs["positive"] >= probs["neutral"]:
                    label = "Positive"
                elif probs["negative"] > probs["positive"] and probs["negative"] >= probs["neutral"]:
                    label = "Negative"
                else:
                    label = "Neutral"
                confidence = max(probs.values())
                return SentimentPrediction(
                    label=label,
                    score=round(net_score, 4),
                    positive_prob=round(probs["positive"], 4),
                    negative_prob=round(probs["negative"], 4),
                    neutral_prob=round(probs["neutral"], 4),
                    confidence=round(confidence, 4),
                    engine=self.engine_name,
                )
            except Exception:
                pass

        if self._nlp_model is not None:
            return self._hybrid_nlp_score(cleaned)

        return self._lexicon_score(cleaned)

    def _hybrid_nlp_score(self, text: str) -> SentimentPrediction:
        """Blend Multilingual TF-IDF Logistic Regression posteriors with contrastive lexicon priors."""
        assert self._nlp_model is not None
        lex_pred = self._lexicon_score(text)

        lower = text.lower()
        split_pos = -1
        for marker in CONTRAST_MARKERS:
            idx = lower.find(marker)
            if idx > 4:
                split_pos = idx
                break

        if split_pos > 0:
            head_probs = self._nlp_model.predict_proba(text[:split_pos])
            tail_probs = self._nlp_model.predict_proba(text[split_pos:])
            nb_probs = {
                c: 0.30 * head_probs[c] + 0.70 * tail_probs[c]
                for c in MultilingualFinancialNLPModel.CLASSES
            }
            lex_tail = self._lexicon_score(text[split_pos:])
            lex_pos = 0.25 * lex_pred.positive_prob + 0.75 * lex_tail.positive_prob
            lex_neg = 0.25 * lex_pred.negative_prob + 0.75 * lex_tail.negative_prob
            lex_neu = 0.25 * lex_pred.neutral_prob + 0.75 * lex_tail.neutral_prob
        else:
            nb_probs = self._nlp_model.predict_proba(text)
            lex_pos, lex_neg, lex_neu = lex_pred.positive_prob, lex_pred.negative_prob, lex_pred.neutral_prob

        pos_prob = 0.60 * nb_probs["Positive"] + 0.40 * lex_pos
        neg_prob = 0.60 * nb_probs["Negative"] + 0.40 * lex_neg
        neu_prob = 0.60 * nb_probs["Neutral"] + 0.40 * lex_neu
        norm = (pos_prob + neg_prob + neu_prob) or 1.0
        pos_prob, neg_prob, neu_prob = pos_prob / norm, neg_prob / norm, neu_prob / norm

        net_score = max(-1.0, min(1.0, (pos_prob - neg_prob) * 1.2))
        if pos_prob >= neg_prob and pos_prob >= neu_prob and net_score > 0.08:
            label = "Positive"
        elif neg_prob > pos_prob and neg_prob >= neu_prob and net_score < -0.08:
            label = "Negative"
        else:
            label = "Neutral"

        confidence = min(0.98, max(pos_prob, neg_prob, neu_prob))
        return SentimentPrediction(
            label=label,
            score=round(net_score, 4),
            positive_prob=round(pos_prob, 4),
            negative_prob=round(neg_prob, 4),
            neutral_prob=round(neu_prob, 4),
            confidence=round(confidence, 4),
            engine=self.engine_name,
        )

    def _lexicon_score(self, text: str) -> SentimentPrediction:
        lower = text.lower()
        tokens = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+|[\uac00-\ud7af]+", lower)

        pos_hits = 0.0
        neg_hits = 0.0

        for idx, tok in enumerate(tokens):
            negated = (idx > 0 and tokens[idx - 1] in NEGATIONS) or (idx > 1 and tokens[idx - 2] in NEGATIONS)
            if any(tok.startswith(w) for w in EN_POS):
                if negated:
                    neg_hits += 0.8
                else:
                    pos_hits += 1.0
            elif any(tok.startswith(w) for w in EN_NEG):
                if negated:
                    pos_hits += 0.6
                else:
                    neg_hits += 1.0

        for phrase in ZH_POS + KO_POS:
            pos_idx = text.find(phrase)
            if pos_idx >= 0:
                prefix = text[max(0, pos_idx - 3) : pos_idx]
                if any(neg in prefix for neg in ("不", "沒", "未", "안", "못")):
                    neg_hits += 0.9
                else:
                    pos_hits += 1.2
        for phrase in ZH_NEG + KO_NEG:
            neg_idx = text.find(phrase)
            if neg_idx >= 0:
                prefix = text[max(0, neg_idx - 3) : neg_idx]
                if any(neg in prefix for neg in ("不", "沒", "未", "안", "못")):
                    pos_hits += 0.7
                else:
                    neg_hits += 1.2

        if re.search(r"\+\s*\d+(\.\d+)?%", text):
            pos_hits += 0.6
        if re.search(r"-\s*\d+(\.\d+)?%", text):
            neg_hits += 0.6

        total = pos_hits + neg_hits
        if total == 0:
            return SentimentPrediction("Neutral", 0.0, 0.18, 0.18, 0.64, 0.60, self.engine_name)

        raw_polarity = (pos_hits - neg_hits) / (total + 1.0)
        score = max(-1.0, min(1.0, math.tanh(raw_polarity * 1.4)))

        if score > 0.12:
            label = "Positive"
        elif score < -0.12:
            label = "Negative"
        else:
            label = "Neutral"

        pos_prob = max(0.05, min(0.92, 0.35 + 0.5 * max(0.0, score) - 0.25 * max(0.0, -score)))
        neg_prob = max(0.05, min(0.92, 0.35 + 0.5 * max(0.0, -score) - 0.25 * max(0.0, score)))
        neu_prob = max(0.05, 1.0 - pos_prob - neg_prob)
        norm = pos_prob + neg_prob + neu_prob
        pos_prob, neg_prob, neu_prob = pos_prob / norm, neg_prob / norm, neu_prob / norm
        confidence = min(0.95, 0.48 + 0.42 * min(1.0, total / 4.0) * (0.5 + 0.5 * abs(score)))

        return SentimentPrediction(
            label=label,
            score=round(score, 4),
            positive_prob=round(pos_prob, 4),
            negative_prob=round(neg_prob, 4),
            neutral_prob=round(neu_prob, 4),
            confidence=round(confidence, 4),
            engine=self.engine_name,
        )

    @staticmethod
    def compute_engagement_weight(likes: int = 0, comments: int = 0, reposts: int = 0, followers: Optional[int] = None) -> float:
        """Compute transparent engagement multiplier: 1 + log1p(likes + 2*comments + 3*reposts)."""
        base = 1.0 + math.log1p(max(0, likes) + 2 * max(0, comments) + 3 * max(0, reposts))
        if followers and followers > 0:
            base *= 1.0 + 0.15 * min(1.0, math.log10(followers) / 6.0)
        return round(base, 4)

    @staticmethod
    def aggregate_scores(items: Sequence[Dict[str, Any]], use_engagement: bool = True) -> Dict[str, Any]:
        total = len(items)
        if total == 0:
            return {
                "total_items": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "sentiment_score": 0.0,
                "engagement_weighted_score": 0.0,
                "bullish_probability": 0.5,
                "bearish_probability": 0.5,
                "confidence_score": 0.0,
            }

        pos = sum(1 for x in items if x.get("sentiment_label") == "Positive")
        neg = sum(1 for x in items if x.get("sentiment_label") == "Negative")
        neu = total - pos - neg

        simple_avg = sum(float(x.get("sentiment_score", 0.0)) for x in items) / total
        weights = [float(x.get("engagement_weight", 1.0)) if use_engagement else 1.0 for x in items]
        w_sum = sum(weights) or 1.0
        weighted_avg = sum(float(x.get("sentiment_score", 0.0)) * w for x, w in zip(items, weights)) / w_sum

        bull_prob = 1.0 / (1.0 + math.exp(-2.6 * weighted_avg))
        bear_prob = 1.0 - bull_prob
        avg_conf = sum(float(x.get("confidence", 0.5)) for x in items) / total

        return {
            "total_items": total,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "positive_pct": round((pos / total) * 100.0, 2),
            "negative_pct": round((neg / total) * 100.0, 2),
            "neutral_pct": round((neu / total) * 100.0, 2),
            "sentiment_score": round(simple_avg, 4),
            "engagement_weighted_score": round(weighted_avg, 4),
            "bullish_probability": round(bull_prob, 4),
            "bearish_probability": round(bear_prob, 4),
            "confidence_score": round(avg_conf, 4),
        }
