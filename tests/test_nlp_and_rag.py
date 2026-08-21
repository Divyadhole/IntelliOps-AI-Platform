"""Module 3 tests: text normalisation, sentiment, topics, retrieval and grounding."""

from __future__ import annotations

import pytest

from intelliops.nlp_engine.clean import clean_corpus, normalise
from intelliops.nlp_engine.sentiment import lexicon_sentiment, score_sentiment
from intelliops.nlp_engine.topics import assign_topics, fit_topics, topic_summary
from intelliops.rag_assistant.assistant import _deterministic_answer, expand_query
from intelliops.rag_assistant.vector_store import Document, VectorStore


class TestCleaning:
    def test_pii_like_tokens_are_masked(self):
        out = normalise("Email me at jane.doe@example.com about order #AB12345 — see https://x.co/abc")
        assert "<email>" in out and "<url>" in out
        assert "jane.doe" not in out

    def test_numbers_are_generalised(self):
        assert "<num>" in normalise("Price went up 24% last month")

    def test_short_documents_are_dropped(self, reviews):
        corpus = reviews.copy()
        corpus.loc[corpus.index[0], "review_text"] = "ok"
        cleaned = clean_corpus(corpus)
        assert (cleaned["token_count"] >= 3).all()


class TestSentiment:
    @pytest.mark.parametrize("text", ["the service is terrible and unusable",
                                      "billing is a mess and support is awful"])
    def test_negative_text_scores_negative(self, text):
        assert lexicon_sentiment(text) < 0

    @pytest.mark.parametrize("text", ["support was excellent and very reliable",
                                      "installation was quick and the value is great"])
    def test_positive_text_scores_positive(self, text):
        assert lexicon_sentiment(text) > 0

    def test_negation_flips_polarity(self):
        assert lexicon_sentiment("this is not great") < lexicon_sentiment("this is great")

    def test_scores_are_bounded(self, reviews, cfg):
        scored = score_sentiment(clean_corpus(reviews)["clean_text"], cfg)
        assert scored["sentiment_score"].between(-1, 1).all()
        assert set(scored["sentiment_label"]) <= {"positive", "negative", "neutral"}

    def test_sentiment_tracks_star_rating(self, reviews, cfg):
        cleaned = clean_corpus(reviews)
        scored = score_sentiment(cleaned["clean_text"], cfg)
        merged = cleaned.assign(score=scored["sentiment_score"].to_numpy())
        low = merged[merged["rating"] <= 2]["score"].mean()
        high = merged[merged["rating"] >= 4]["score"].mean()
        assert high > low, "sentiment must correlate with the star rating it never sees"


class TestTopics:
    @pytest.fixture(scope="class")
    def modelled(self, reviews, cfg):
        cleaned = clean_corpus(reviews)
        model = fit_topics(cleaned["clean_text"], cfg)
        assigned = assign_topics(model, cleaned["clean_text"])
        return cleaned.reset_index(drop=True).join(assigned)

    def test_every_document_gets_a_topic(self, modelled, cfg):
        assert modelled["topic_id"].notna().all()
        assert modelled["topic_id"].nunique() <= int(cfg["nlp.n_topics"])

    def test_labels_are_readable_and_distinct(self, modelled):
        labels = modelled[["topic_id", "topic_label"]].drop_duplicates()
        assert labels["topic_label"].is_unique
        assert not labels["topic_label"].str.contains("topic_").any()

    def test_summary_ranks_by_negativity(self, modelled, cfg):
        from intelliops.nlp_engine.sentiment import score_sentiment

        scored = score_sentiment(modelled["clean_text"], cfg)
        enriched = modelled.join(scored)
        summary = topic_summary(enriched)
        assert summary["negative_share"].is_monotonic_decreasing
        assert summary["documents"].sum() == len(enriched)


class TestVectorStore:
    @pytest.fixture(scope="class")
    def store(self):
        docs = [
            Document("d1", "Fibre customers on month-to-month contracts churn the most.", "kpi"),
            Document("d2", "Billing errors and surprise price increases drive complaints.", "topic_summary"),
            Document("d3", "The delivery of my replacement router was late again.", "review"),
            Document("d4", "Support fixed my issue in one call, excellent service.", "review"),
            Document("d5", "Churn model driver #1: tenure, mean absolute SHAP 0.87.", "model_driver"),
        ]
        return VectorStore(backend="tfidf").build(docs)

    def test_retrieves_the_relevant_document(self, store):
        top = store.search("late delivery of my router", top_k=1)[0][0]
        assert top.doc_id == "d3"

    def test_source_filter_is_respected(self, store):
        hits = store.search("anything", top_k=3, sources=["review"])
        assert {d.source for d, _ in hits} == {"review"}

    def test_stratified_search_covers_every_source(self, store):
        hits = store.search_stratified("why are customers unhappy",
                                       {"kpi": 1, "topic_summary": 1, "review": 2, "model_driver": 1})
        assert {d.source for d, _ in hits} == {"kpi", "topic_summary", "review", "model_driver"}

    def test_results_are_ordered_by_score(self, store):
        scores = [s for _, s in store.search("billing complaints", top_k=4)]
        assert scores == sorted(scores, reverse=True)

    def test_round_trips_through_disk(self, store, tmp_path):
        path = store.save(tmp_path / "store.joblib")
        reloaded = VectorStore.load(path)
        assert len(reloaded.documents) == len(store.documents)
        assert reloaded.search("late delivery", top_k=1)[0][0].doc_id == "d3"


class TestGrounding:
    def test_query_expansion_bridges_the_vocabulary_gap(self):
        expanded = expand_query("why are customers leaving?")
        assert "churn" in expanded and "cancel" in expanded

    def test_expansion_leaves_unrelated_questions_alone(self):
        assert expand_query("what is the tenure distribution") == "what is the tenure distribution"

    def test_deterministic_answer_cites_every_claim(self):
        hits = [
            (Document("k1", "Churn rate is 26.5%.", "kpi", {}), 0.9),
            (Document("t1", "Billing theme: 80% negative.", "topic_summary", {"negative_share": 0.8}), 0.8),
        ]
        answer = _deterministic_answer("why are customers leaving?", hits)
        assert "[E1]" in answer and "[E2]" in answer
        assert "Confidence & caveats" in answer
