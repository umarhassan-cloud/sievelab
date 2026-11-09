import json

from dataforge.eval.ngram_lm import CharKneserNeyLM
from dataforge.pipeline import Pipeline, load_config
from dataforge.pipeline.config import apply_overrides
from dataforge.pipeline.io import write_jsonl
from dataforge.synth import CorpusGenerator


def small_config(tmp_path, **over):
    gen = CorpusGenerator(seed=5)
    write_jsonl(tmp_path / "corpus.jsonl", gen.crawl_corpus(n_docs=300))
    write_jsonl(tmp_path / "qtrain.jsonl", CorpusGenerator(seed=6).quality_train_set(300))
    cfg = load_config(None, [
        f"inputs=['{(tmp_path / 'corpus.jsonl').as_posix()}']",
        f"output_dir={(tmp_path / 'out').as_posix()}",
        f"quality.model_path={(tmp_path / 'm.pkl').as_posix()}",
        f"quality.train_data={(tmp_path / 'qtrain.jsonl').as_posix()}",
        "heuristics.min_alpha_word_frac=0.6",
    ])
    return apply_overrides(cfg, [f"{k}={v}" for k, v in over.items()])


def test_end_to_end_pipeline(tmp_path):
    cfg = small_config(tmp_path)
    res = Pipeline(cfg, tmp_path).run()
    m = res.manifest
    names = [s.name for s in m.stages]
    assert names == ["extract", "lang_filter", "heuristic_filter", "dedup", "quality_filter", "pii", "toxicity"]
    for a, b in zip(m.stages, m.stages[1:]):
        assert b.n_in == a.n_out
    assert 0 < m.n_final < m.n_ingested
    assert all(d.lang in ("en", "ur") for d in res.documents)
    out = tmp_path / "out"
    for f in ("clean.jsonl", "clean.parquet", "rejected.jsonl", "manifest.json", "manifest.md", "eval.json"):
        assert (out / f).exists(), f
    exported = [json.loads(line) for line in (out / "clean.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(exported) == m.n_final
    assert all("_gold" not in d["meta"] for d in exported)
    ev = json.loads((out / "eval.json").read_text(encoding="utf-8"))
    assert ev["dedup"]["recall_on_injected"] > 0.8
    assert ev["langid"]["accuracy"] > 0.95


def test_pii_is_redacted_in_output(tmp_path):
    res = Pipeline(small_config(tmp_path), tmp_path).run(export=False)
    for d in res.documents:
        assert "@example" not in d.text


def test_toxicity_redact_policy_keeps_documents(tmp_path):
    drop = Pipeline(small_config(tmp_path), tmp_path).run(export=False)
    red = Pipeline(small_config(tmp_path, **{"safety.toxicity.action": "redact_sentences"}), tmp_path).run(export=False)
    assert red.manifest.n_final >= drop.manifest.n_final


def test_kneser_ney_normalises():
    lm = CharKneserNeyLM(order=3).fit(["abracadabra", "banana bandana"])
    for ctx in ("ab", "an", "zz", "\x02\x02"):
        total = sum(lm.prob(ctx, c) for c in lm.vocab) + lm.prob(ctx, "\uffff")  # + one unseen symbol
        assert abs(total - 1) < 1e-9, ctx
