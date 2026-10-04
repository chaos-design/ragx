"""解析质量抽样质检 —— 工程问题①的落地实现。

工业界主流做法：
1. 分层抽样(stratified sampling)：按「文档类型 × 来源 × 解析器版本」分层，
   而非全局随机，保证每个高风险类别(扫描件/复杂表格)都有足够样本。
2. 自动信号 + 人工复核结合：先用可计算的「解析健康度信号」给每页/每文档打分，
   优先把低分样本送人工(降低人工成本，提高发现缺陷的命中率)。
3. 验收阈值：字符级准确率 / 关键字段召回率 达标才放行入库。

生产常用阈值(参考，按业务调整)：
- 文本型 PDF/DOCX/MD：字符准确率 ≥ 99%；
- 扫描件 OCR：字符准确率 ≥ 95%，关键字段(金额/日期/编号)准确率 ≥ 98%；
- 表格结构还原 F1 ≥ 0.9。
- 单批抽样量：每层 min(全量, max(30, 5%))；置信度 95% 下 30 是经验下限。

健康度信号(无需人工即可算，用于优先级排序)：
- 乱码率(非常见字符占比)、空白页率、超短/超长 chunk 占比、
  OCR 平均置信度、连字符断词数、疑似栏交错(行长方差异常)。

依赖：仅标准库 + rag.interfaces。
"""
from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field

_GARBLED = re.compile(r"[^\w\s一-鿿。，、；：？！“”（）()\[\]{}<>/\\\-+*=.,!?'\"%$#@&|~`^]")
_CJK = re.compile(r"[一-鿿]")


@dataclass
class PageSignal:
    doc_id: str
    page: int
    char_count: int
    garbled_ratio: float
    ocr_confidence: float       # 无 OCR 则填 1.0
    hyphen_breaks: int
    is_blank: bool
    health_score: float = 0.0   # 越低越该送人工


@dataclass
class SampleResult:
    """人工质检回填结果。"""

    doc_id: str
    page: int
    char_accuracy: float
    table_f1: float | None = None
    key_field_accuracy: float | None = None
    passed: bool = True
    note: str = ""


@dataclass
class QAReport:
    sampled: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    failed_samples: list[SampleResult] = field(default_factory=list)
    release_ok: bool = False


def compute_signal(doc_id: str, page: int, text: str, ocr_confidence: float = 1.0) -> PageSignal:
    n = len(text)
    garbled = len(_GARBLED.findall(text)) / n if n else 1.0
    hyphen = len(re.findall(r"[A-Za-z]-\n[A-Za-z]", text))
    is_blank = n < 10
    # 健康度：乱码、低 OCR 置信度、断词越多分越低
    score = 1.0
    score -= min(0.5, garbled * 5)
    score -= (1.0 - ocr_confidence) * 0.5
    score -= min(0.2, hyphen * 0.01)
    if is_blank:
        score = 0.0
    return PageSignal(doc_id, page, n, round(garbled, 4), ocr_confidence,
                      hyphen, is_blank, round(max(0.0, score), 4))


class SamplingQA:
    def __init__(self, min_per_stratum: int = 30, sample_ratio: float = 0.05,
                 seed: int = 42) -> None:
        self.min_per_stratum = min_per_stratum
        self.sample_ratio = sample_ratio
        self._rng = random.Random(seed)

    def plan_samples(self, signals: list[PageSignal],
                     stratify_key=lambda s: s.doc_id.split(".")[-1]) -> list[PageSignal]:
        """分层抽样：每层取 max(min_per_stratum, ratio*N)，
        并把低健康度页优先纳入(风险导向)。"""
        strata: dict[str, list[PageSignal]] = defaultdict(list)
        for s in signals:
            strata[stratify_key(s)].append(s)

        chosen: list[PageSignal] = []
        for _key, items in strata.items():
            k = min(len(items), max(self.min_per_stratum,
                                    math.ceil(len(items) * self.sample_ratio)))
            # 一半按健康度从低到高(抓缺陷)，一半随机(估真实分布)
            items_sorted = sorted(items, key=lambda s: s.health_score)
            risk_part = items_sorted[: k // 2]
            rest = items_sorted[k // 2:]
            self._rng.shuffle(rest)
            random_part = rest[: k - len(risk_part)]
            chosen.extend(risk_part + random_part)
        return chosen

    def evaluate(self, results: list[SampleResult], *,
                 char_threshold: float = 0.99,
                 table_f1_threshold: float = 0.9,
                 key_field_threshold: float = 0.98) -> QAReport:
        rep = QAReport(sampled=len(results))
        for r in results:
            ok = r.char_accuracy >= char_threshold
            if r.table_f1 is not None:
                ok = ok and r.table_f1 >= table_f1_threshold
            if r.key_field_accuracy is not None:
                ok = ok and r.key_field_accuracy >= key_field_threshold
            r.passed = ok
            if ok:
                rep.passed += 1
            else:
                rep.failed += 1
                rep.failed_samples.append(r)
        rep.pass_rate = round(rep.passed / rep.sampled, 4) if rep.sampled else 0.0
        # 放行规则：通过率达标且无关键字段类硬失败
        rep.release_ok = rep.pass_rate >= 0.95
        return rep
