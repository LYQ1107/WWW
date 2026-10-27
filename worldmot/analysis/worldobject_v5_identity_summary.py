"""Render already frozen all110 diagnostics; never open dataset or predictions."""
from collections import Counter
import csv
import io
import math
from pathlib import Path

from worldmot.runtime.world_object_v5_contract import atomic_json, FULL_SHA, LEGAL_TEST
from worldmot.data.worldobject_cache import sha256

VARIANTS = ("B0", "B1", "B2", "B3", "B4")
GROUPS = ("legal14", "invalid8_diagnostic")
ANNOTATIONS = "7a1e735bf25026a56148deaeb432e0520c6ad5dad97d5362aa58739255389593"
INITIAL = "3fb3f3d6336e8aa32d635ecff3c05929b0d4b8f265f4a0ccf56771814d13ed23"
PENDING = "PENDING_ALL110_IDENTITY_ATTRIBUTION"
COMPLETE = "COMPLETE_FROZEN_IDENTITY_ANALYSIS_SUMMARY"
METRICS = ("HOTA", "AssA", "IDF1", "CVIDF1", "MOTA", "IDSW")
RATE_SPECS = (
    ("output_coverage", "output_candidates", "frozen_candidates", 100., "percent"),
    ("clean_output_coverage", "clean_labelled_candidates_retained", "clean_labelled_frozen_candidates", 100., "percent"),
    ("same_person_shared_ID", "correct_same_person_shared_ID", "same_person_cross_view_pairs", 100., "percent"),
    ("different_person_false_merge", "false_different_person_shared_ID", "different_person_cross_view_pairs", 100., "percent"),
    ("false_birth_proxy", "false_birth_unique_active_history_observations", "clean_labelled_candidates_retained", 1000., "per_1000_clean_retained_observations"),
    ("wrong_historical_reuse", "wrong_historical_ID_reuse", "clean_labelled_candidates_retained", 1000., "per_1000_clean_retained_observations"),
    ("NULL_despite_current_person", "NULL_despite_current_person", "NULL_unique_active_history_opportunities", 100., "percent"),
    ("LOST_pure_identity_recovery", "LOST_recovery_correct_pure_identity", "LOST_recovery_clean_observations", 100., "percent"),
    ("gap_2to30_ID_retained", "detection_gap_retained_ID_2to30", "detection_gap_returns_2to30", 100., "percent"),
    ("gap_over30_ID_retained", "detection_gap_retained_ID_over30", "detection_gap_returns_over30", 100., "percent"),
    ("cumulatively_mixed_ID_output", "clean_observations_into_cumulatively_mixed_ID", "clean_labelled_candidates_retained", 100., "percent"),
    ("diagnostic_temporal_ID_change", "diagnostic_per_camera_ID_changes", "clean_per_camera_temporal_links", 100., "percent"),
    ("wrong_reuse_on_currently_written_ID", "wrong_person_identity_write_observations", "identity_write_timestamp_available_observations", 1000., "per_1000_observations_with_write_timestamp"),
)


def count(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("NONNEGATIVE_INTEGER_DIAGNOSTIC_COUNTS_REQUIRED")
    return value


def rates(counts):
    rows = []
    for event, numerator, denominator, scale, unit in RATE_SPECS:
        n, d = count(counts.get(numerator, 0)), count(counts.get(denominator, 0))
        if n > d:
            raise ValueError("EVENT_NUMERATOR_EXCEEDS_ITS_DECLARED_DENOMINATOR")
        rows.append({"event": event, "numerator_key": numerator, "denominator_key": denominator,
                     "numerator": n, "denominator": d, "value": scale * n / d if d else None,
                     "unit": unit})
    return rows


def contrast_row(scope, contrast, stratum, event, counts):
    n = count(counts.get("opportunities", 0))
    corrected, regressed = count(counts.get("corrections", 0)), count(counts.get("regressions", 0))
    net = counts.get("net", 0)
    if (not isinstance(net, int) or isinstance(net, bool) or net != corrected - regressed
            or corrected + regressed > n):
        raise ValueError("PAIRED_CORRECTIONS_REGRESSIONS_NET_OR_OPPORTUNITIES_INCONSISTENT")
    return {"scope": scope, "contrast": contrast, "stratum": stratum, "event": event,
            "opportunities": n, "corrections": corrected, "regressions": regressed, "net": net,
            "net_per_1000_opportunities": 1000. * net / n if n else None}


def paired_rows(data):
    rows = []
    for group in GROUPS:
        for key, values in sorted(data["paired"][group].items()):
            contrast, event = key.split("/", 1)
            if event == "detection_temporal_events":
                kinds = {k.rsplit("/", 1)[0] for k in values}
                for kind in sorted(kinds):
                    counts = {name: values.get(kind + "/" + name, 0)
                              for name in ("opportunities", "corrections", "regressions", "net")}
                    rows.append(contrast_row(group, contrast, "all", kind, counts))
            else:
                rows.append(contrast_row(group, contrast, "all", event, values))
    for key, values in sorted(data["legal14_shared_input_strata"].items()):
        contrast, stratum, event = key.split("/", 2)
        rows.append(contrast_row("legal14", contrast, stratum, event, values))
    return rows


def validate_inputs(data, official, gates, common):
    if (data.get("status") != "COMPLETE_FROZEN_ALL5_FULL22_IDENTITY_ATTRIBUTION"
            or data.get("official_metric") is not False or data.get("TEST_parameter_selection") is not False
            or data.get("new_inference_or_training_executed") is not False
            or data.get("GT_role") != "POSTHOC_ONLY_AFTER_ALL110_OUTPUTS_FROZEN"
            or data.get("original_GT_sha256") != ANNOTATIONS or data.get("legal14_scenes") != LEGAL_TEST
            or official.get("status") != "COMPLETE_NATIVE_LEGAL14_ALL5"
            or official.get("canonical_TEST_scope_complete_variants") != list(VARIANTS)
            or official.get("raw_GT_modified") is not False or official.get("TEST_parameter_selection") is not False
            or data.get("v5_source_sha256") != official.get("v5_source_sha256")
            or set(official["V5_official_metrics"]) != set(VARIANTS)
            or set(data["scenes"]) != {str(s) for s in range(1, 23)}):
        raise ValueError("COMPLETE_ALL110_NATIVE_LEGAL14_ORIGINAL_GT_SCOPE_REQUIRED")
    if (not common or common.get("status") != "FROZEN_TRAIN_ONLY_BUDGET"
            or common.get("epochs") not in (12, 16) or common.get("TEST_read") is not False
            or common.get("v5_source_sha256") != data["v5_source_sha256"]):
        raise ValueError("UNCHANGED_TRAIN_ONLY_COMMON_BUDGET_REQUIRED")
    for variant in ("B2", "B3", "B4"):
        gate = gates.get(variant) or {}
        coverage = gate.get("final_epoch_coverage", [])
        if (gate.get("status") != "FULL_TRAIN_GATE_PASSED" or gate.get("variant") != variant
                or gate.get("v5_source_sha256") != data["v5_source_sha256"]
                or gate.get("final_epochs") != common["epochs"] or gate.get("trainable_parameters") != 606468
                or gate.get("seed") != 20261027 or gate.get("history_mode") != "predicted_from_first_timestamp"
                or gate.get("clip_identity_reset") is not False
                or gate.get("initial_state_tensor_sha256") != INITIAL
                or gate.get("checkpoint_sha256") != official["checkpoint_SHA256"].get(variant)
                or gate.get("final_scenes") != list(range(1, 25)) or gate.get("final_timestamps") != 28754
                or gate.get("final_total_training_timestamps") != 28754 * common["epochs"]
                or gate.get("GT_in_model_inputs") is not False or gate.get("GT_repair_of_predicted_history") is not False
                or len(coverage) != common["epochs"]
                or any(sorted(x["scene"] for x in epoch) != list(range(1, 25))
                       or sum(x["timestamps"] for x in epoch) != 28754 for epoch in coverage)):
            raise ValueError("ACTUAL_MATCHED_FULL24_TRAIN_GATES_REQUIRED")
    expected_totals = {group: {v: Counter() for v in VARIANTS} for group in GROUPS}
    for scene, record in data["scenes"].items():
        group = "legal14" if int(scene) in LEGAL_TEST else "invalid8_diagnostic"
        if record["original_GT_legal"] != (group == "legal14"):
            raise ValueError("INVALID8_MUST_REMAIN_SEPARATE_DIAGNOSTICS")
        for variant in VARIANTS:
            quarters = record["relative_video_quarters"][variant]
            if len(quarters) != 4:
                raise ValueError("FOUR_RELATIVE_VIDEO_QUARTERS_REQUIRED")
            merged = Counter()
            for quarter in quarters:
                merged.update({k: count(v) for k, v in quarter.items()})
            counts = {k: count(v) for k, v in record["counts"][variant].items()}
            if any(merged[k] != counts.get(k, 0) for k in merged.keys() | counts.keys()):
                raise ValueError("QUARTER_EXPOSURES_MUST_SUM_TO_COMPLETE_SCENE")
            expected_totals[group][variant].update(counts)
    for variant in VARIANTS:
        proof = data["proofs"][variant]
        native = {x["scene"]: x for x in official["proofs"][variant]["coverage"]}
        if (sorted(x["scene"] for x in proof) != list(range(1, 23))
                or set(native) != set(range(1, 23))
                or sum(x["timestamps"] for x in proof) != 29019
                or sum(x["candidates"] for x in proof) != 548583
                or any(x["tracks_sha256"] != native[x["scene"]]["predictions_sha256"]
                       or x["timestamps"] != native[x["scene"]]["frames"] for x in proof)):
            raise ValueError("ATTRIBUTION_AND_NATIVE_SCORES_MUST_USE_THE_SAME_ALL22_PREDICTIONS")
        for group in GROUPS:
            expected, actual = expected_totals[group][variant], data["totals"][group][variant]
            if any(expected[k] != actual.get(k, 0) for k in expected.keys() | actual.keys()):
                raise ValueError("SCENE_COUNTS_MUST_SUM_TO_THE_DECLARED_LEGAL_OR_INVALID_GROUP")
        if sum(data["totals"][g][variant].get("frozen_candidates", 0) for g in GROUPS) != 548583:
            raise ValueError("COMMON_RAW_CANDIDATE_DENOMINATOR_REQUIRED")


def build_identity_summary(data, official, gates, common, birth):
    if data is None:
        return {"status": PENDING, "formal_MOT_metric": False, "rates": [], "contrasts": [],
                "quarters": [], "complex_scenes": [], "reason": "All110 frozen predictions and native14 scoring are required."}
    validate_inputs(data, official, gates, common)
    rows, quarters, complex_scenes = [], [], []
    for group in GROUPS:
        for variant in VARIANTS:
            rows.extend({"scope": group, "method": variant, **r} for r in rates(data["totals"][group][variant]))
    for scene, record in sorted(data["scenes"].items(), key=lambda x: int(x[0])):
        scope = "legal14" if record["original_GT_legal"] else "invalid8_diagnostic"
        for variant in VARIANTS:
            for quarter, counts in enumerate(record["relative_video_quarters"][variant], 1):
                quarters.extend({"scope": scope, "scene": int(scene), "method": variant,
                                 "quarter": quarter, **r} for r in rates(counts))
        if scope == "legal14":
            values = {r["event"]: r["value"] for r in rates(record["counts"]["B3"])}
            complex_scenes.append({"scene": int(scene),
                "clean_retained_observations": record["counts"]["B3"].get("clean_labelled_candidates_retained", 0),
                **{k: values[k] for k in ("cumulatively_mixed_ID_output", "wrong_historical_reuse",
                                         "false_birth_proxy", "LOST_pure_identity_recovery")}})
    comparisons = (("B0", "B1"), ("B1", "B2"), ("B2", "B3"), ("B4", "B3"), ("B0", "B3"))
    values = official["V5_official_metrics"]
    formal_deltas = {right + "_minus_" + left: {k: values[right][k] - values[left][k] for k in METRICS}
                     for left, right in comparisons}
    references = {name: {k: values["B3"][k] - row[k] for k in METRICS}
                  for name, row in official["historical_baselines"].items()}
    worst_native = []
    for name, row in official["proofs"]["B3"]["per_scene_TrackEval"].items():
        cv = official["proofs"]["B3"]["per_scene_MATLAB"][name]["CVIDF1"]
        old_cv = official["proofs"]["B4"]["per_scene_MATLAB"][name]["CVIDF1"]
        worst_native.append({"native_scene_name": name, "HOTA": row["HOTA"], "AssA": row["AssA"],
                             "IDF1": row["IDF1"], "CVIDF1": cv, "delta_CVIDF1_vs_B4": cv - old_cv})
    contrasts = paired_rows(data)
    for row in contrasts:
        if row["scope"] == "legal14" and row["stratum"] == "all" and row["event"] in (
                "same_person_shared_ID", "different_person_separate_ID"):
            strata = [r for r in contrasts if r["scope"] == "legal14" and r["contrast"] == row["contrast"]
                      and r["event"] == row["event"] and r["stratum"] in ("geometry_reliable", "geometry_unreliable")]
            if len(strata) != 2 or any(sum(x[k] for x in strata) != row[k]
                                      for k in ("opportunities", "corrections", "regressions", "net")):
                raise ValueError("SHARED_INPUT_GEOMETRY_STRATA_MUST_PARTITION_THE_LEGAL14_PAIRS")
    return {"status": COMPLETE, "formal_MOT_metric": False, "rates": rows, "contrasts": contrasts,
        "quarters": quarters, "complex_scenes": sorted(complex_scenes,
            key=lambda x: x["cumulatively_mixed_ID_output"] if x["cumulatively_mixed_ID_output"] is not None else -1, reverse=True),
        "worst_native_scenes": sorted(worst_native, key=lambda x: x["CVIDF1"]),
        "native_deltas": formal_deltas, "B3_minus_historical": references,
        "B3_exceeds_GMT_three_primary_metrics": all(references["GMT Native"][k] > 0 for k in ("HOTA", "AssA", "CVIDF1")),
        "B3_vs_B4_three_primary_metrics_positive": all(formal_deltas["B3_minus_B4"][k] > 0 for k in ("HOTA", "AssA", "CVIDF1")),
        "same_capacity_parameters": 606468, "same_final_epochs": common["epochs"],
        "checkpoint_sha256": {"B0": FULL_SHA, "B1": FULL_SHA, **official["checkpoint_SHA256"]},
        "v5_source_sha256": data["v5_source_sha256"], "original_GT_sha256": data["original_GT_sha256"],
        "definitions": data["definitions"], "TRAIN_conditional_birth": birth.get("versus_original", {}),
        "false_birth_is_an_outcome_proxy": True, "mixed_ID_is_not_a_direct_embedding_measurement": True,
        "write_coincidence_identifies_the_ID_timestamp_not_the_originating_observation": True,
        "diagnostic_gaps_do_not_prove_occlusion": True, "original_GT_modified": False,
        "TEST_parameter_selection": False, "new_inference_or_training_executed": False}


def fmt(value, signed=False):
    if value is None:
        return "N/A"
    if not math.isfinite(value):
        raise ValueError("NONFINITE_REPORT_VALUE")
    return format(value, "+.4f" if signed else ".4f")


def render_identity_analysis(summary, live=None):
    lines = ["WorldMOT V5：完整 TEST 身份关联分析", ""]
    if summary["status"] != COMPLETE:
        return "\n".join(lines + ["状态：等待全部110份冻结预测、正式14场评分和完整身份归因。",
            "当前不输出恢复率、净纠错数或几何收益结论；空表不代表零错误。", ""])
    lines += ["正式指标采用原始合法14场的 TrackEval 和 VisionTrack MATLAB，全部22场均完整推理。",
        "下面的事件统计使用固定的干净 IoU=.5 标注连接，仅作事后诊断；不是官方 IDSW 或 CVIDF1。异常8场单列在 CSV/JSON 中。",
        "", "分离 Birth 的完整系统对照为 B1−B0；它还改变历史匹配、生命周期和写入策略，不能把全部变化归因于 Birth。"
        "TRAIN 条件归因保持原 C1 历史不变，其结果只解释当前条件下的出生决策。",
        "", "| TRAIN 条件对照 | 同人共享纠正 | 回归 | 净变化 |", "|---|---:|---:|---:|"]
    for name, counts in summary["TRAIN_conditional_birth"].items():
        lines.append(f"| {name} | {counts.get('same_person_shared/corrections', 0)} | "
                     f"{counts.get('same_person_shared/regressions', 0)} | {counts.get('same_person_shared/net', 0):+d} |")
    lines += ["", "专用身份头的 B2−B1 对照同时改变模型容量和训练预算，表示重新训练后的系统变化，不能单独证明架构效应。"
        f"B3−B4 保持606,468个参数和相同的{summary['same_final_epochs']}轮全24场训练，比较显式 Ω 世界支持、状态和关联路径的整体贡献。",
        "", "| 正式对照 | ΔHOTA | ΔAssA | ΔIDF1 | ΔCVIDF1 | ΔMOTA | ΔIDSW |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary["native_deltas"].items():
        lines.append("| " + name + " | " + " | ".join(
            format(int(row[k]), "+d") if k == "IDSW" else fmt(row[k], True) for k in METRICS) + " |")
    birth_gain = all(summary["native_deltas"]["B1_minus_B0"][k] > 0 for k in ("HOTA", "AssA", "CVIDF1"))
    head_gain = all(summary["native_deltas"]["B2_minus_B1"][k] > 0 for k in ("HOTA", "AssA", "CVIDF1"))
    lines += ["", "冻结旧身份头下的完整策略替换：" + ("三个主指标均有数值提高。" if birth_gain else "没有三个主指标一致正向的证据。") +
              "它不能单独证明 Birth 的因果收益。",
              "专用头重新训练后的系统对照：" + ("三个主指标均提高。" if head_gain else "未在三个主指标上全部提高。") +
              "容量和预算不同，不能将差值全部归因于头部结构。"]
    lines += ["", "B3−B4 三个主指标是否全部正向：" + ("是。" if summary["B3_vs_B4_three_primary_metrics_positive"] else "否，不能宣称三个主指标一致受益。"),
        "B3 是否在 HOTA、AssA、CVIDF1 上全部超过 GMT：" + ("是，本次单 seed 结果达到数值目标。" if summary["B3_exceeds_GMT_three_primary_metrics"] else "否，本次实验没有达到全部超过 GMT 的科学目标。"),
        "", "| 历史参考：B3 减参考 | ΔHOTA | ΔAssA | ΔIDF1 | ΔCVIDF1 | ΔMOTA | ΔIDSW |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary["B3_minus_historical"].items():
        lines.append("| " + name + " | " + " | ".join(
            format(int(row[k]), "+d") if k == "IDSW" else fmt(row[k], True) for k in METRICS) + " |")
    lines += ["", "百分制指标差值的单位是百分点，IDSW 差值的单位是事件数。",
        "历史 V2 的检测/输出规则不同，相关差值是系统参考；历史 GMT/V2 使用两位小数坐标导出，当前使用 .17g。"
        "正的 DetA 或少量 TP 差异可能受身份条件匹配影响，不能单据此宣称检测器改善。",
        "", "事件比例保留各自分子和机会数。False Birth 按有唯一纯净活动历史时仍生成新 ID 的观测计数，下面按每千个保留的干净观测归一化。"
        "LOST 恢复率仅以真正复用先前 LOST ID 的干净观测为分母；无机会时显示 N/A。",
        "", "| 方法：合法14场诊断 | 输出覆盖% | 同人共享% | 异人误融合% | False Birth/千观测 | 错误历史复用/千观测 | LOST正确恢复% | 累计混合ID输出% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        values = {r["event"]: r["value"] for r in summary["rates"] if r["scope"] == "legal14" and r["method"] == variant}
        keys = ("output_coverage", "same_person_shared_ID", "different_person_false_merge", "false_birth_proxy",
                "wrong_historical_reuse", "LOST_pure_identity_recovery", "cumulatively_mixed_ID_output")
        lines.append("| " + variant + " | " + " | ".join(fmt(values[k]) for k in keys) + " |")
    lines += ["", "累计混合 ID 表示同一个输出 ID 曾关联多个干净 GT 身份，不直接测量内部特征污染。"
        "写入时间戳统计只说明错误复用的 ID 在该时刻发生过写入，不能识别具体是哪一路观测写入。"
        "检测间隔可能来自遮挡、出视野或缺标，不能统称遮挡恢复。",
        "LOST 指标是已经复用旧 LOST ID 时的纯净身份正确率，不是所有可恢复目标的召回率。下面独立报告检测间隔的保 ID 比例。",
        "", "| 方法：合法14场诊断 | NULL% | 2–30帧间隔保ID% | 更长间隔保ID% | 诊断时序ID改变% |",
        "|---|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        values = {r["event"]: r["value"] for r in summary["rates"] if r["scope"] == "legal14" and r["method"] == variant}
        keys = ("NULL_despite_current_person", "gap_2to30_ID_retained", "gap_over30_ID_retained", "diagnostic_temporal_ID_change")
        lines.append("| " + variant + " | " + " | ".join(fmt(values[k]) for k in keys) + " |")
    lines += [
        "", "| Ω 独立对照：B3−B4 | 事件 | 机会数 | 纠正 | 回归 | 净变化 | 每千机会净变化 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["contrasts"]:
        if row["scope"] == "legal14" and row["contrast"] == "B3_minus_B4" and row["stratum"] in ("all", "geometry_reliable", "geometry_unreliable"):
            lines.append(f"| {row['stratum']} | {row['event']} | {row['opportunities']} | {row['corrections']} | "
                         f"{row['regressions']} | {row['net']:+d} | {fmt(row['net_per_1000_opportunities'], True)} |")
    lines += ["", "几何分层对所有方法使用相同的 B3 冻结输入标记。可靠性是输入质量/对齐判据，不能视为真实3D GT。"
        "继承缓存大部分时刻缺少世界与 Ω 相机坐标之间的变换，因此这不是完整重投影模块收益的证明。",
        "", "| B3 累计混合输出比例最高的合法场景 | 干净保留观测 | 混合ID输出% | 错误复用/千观测 | False Birth/千观测 |",
        "|---|---:|---:|---:|---:|"]
    for row in summary["complex_scenes"][:5]:
        lines.append(f"| Scene {row['scene']} | {row['clean_retained_observations']} | {fmt(row['cumulatively_mixed_ID_output'])} | "
                     f"{fmt(row['wrong_historical_reuse'])} | {fmt(row['false_birth_proxy'])} |")
    lines += ["", "| B3 CVIDF1 最低的原生场景 | HOTA | AssA | IDF1 | CVIDF1 | ΔCVIDF1 对 B4 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summary["worst_native_scenes"][:5]:
        lines.append(f"| {row['native_scene_name']} | {fmt(row['HOTA'])} | {fmt(row['AssA'])} | "
                     f"{fmt(row['IDF1'])} | {fmt(row['CVIDF1'])} | {fmt(row['delta_CVIDF1_vs_B4'], True)} |")
    lines += ["", "长视频按每个场景的相对四分段统计，先汇总分子/分母再求比例；不是逐场景比例的简单平均。"
        "累计污染标记本身保留历史，分段上升不能单独证明新的特征写入造成污染。",
        "", "| 方法 | 视频四分段 | 混合ID输出分子 | 干净保留分母 | 混合ID输出% |", "|---|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        for quarter in range(1, 5):
            rows = [r for r in summary["quarters"] if r["scope"] == "legal14" and r["method"] == variant
                    and r["quarter"] == quarter and r["event"] == "cumulatively_mixed_ID_output"]
            n, d = sum(r["numerator"] for r in rows), sum(r["denominator"] for r in rows)
            lines.append(f"| {variant} | {quarter} | {n} | {d} | {fmt(100. * n / d if d else None)} |")
    lines += ["", "实际在线速度见 v5_runtime.csv 和 v5_runtime.json。缓存步速/输出存储条件与真实 RGB→ID 速度分开；"
        "真实计时范围是 TRAIN4/8/12 各61个同步时刻，共183组，含原图读取、检测、每帧Ω（B4除外）、关联和最终ID交付。"
        "它不是全部 TEST 的原图端到端 FPS，也不是多 GPU 聚合吞吐。",
        "", "| 方法 | 真实 RGB→ID FPS（同步双摄时刻/秒） |", "|---|---:|"]
    for variant in VARIANTS:
        item = (live or {}).get("variants", {}).get(variant, {})
        value = item.get("LIVE_RGB_to_final_ID_FPS") if (live or {}).get("status") == "ALL5_FINITE_REAL_LIVE_COMPLETE" else None
        lines.append(f"| {variant} | {fmt(value)} |")
    if (live or {}).get("status") == "ALL5_FINITE_REAL_LIVE_COMPLETE":
        measured = live["variants"]
        first, second = measured["B3"]["LIVE_RGB_to_final_ID_FPS"], measured["B4"]["LIVE_RGB_to_final_ID_FPS"]
        if first <= 0 or second <= 0 or not math.isfinite(first + second):
            raise ValueError("POSITIVE_FINITE_REAL_MEASURED_FPS_REQUIRED")
        lines += ["", f"B3 的真实采样 FPS 为 B4 的 {first / second:.4f} 倍；这是同范围的速度代价/收益，不能替代身份质量比较。"]
    lines += ["", "此速度协议没有同范围的新 GMT Native 原图计时，不能据此声称比 GMT 更快；"
              "协议也没有注册一个固定的实时 FPS 门槛。",
              "Ω 主线的三个匹配主指标" + ("均有正向结果，仍需结合上述实测速度取舍。" if
              summary["B3_vs_B4_three_primary_metrics_positive"] else
              "没有全部改善，当前缺少把 Ω 版本确立为默认主线的一致指标证据。")]
    lines += ["", "结果来自本次固定单 seed 协议，不能据此声称多 seed 稳定性或统计显著性。"
        "完整机会数、分子、分母、全部对照及异常8场诊断分别保存在 JSON 和 CSV 中；具体失败帧保留在 v5_failure_cases.json。", ""]
    return "\n".join(lines)


def save_csv(path, columns, rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(output.getvalue())
    temporary.replace(path)


def write_identity_summary(summary, results, docs, live=None):
    results, docs = Path(results), Path(docs)
    save_csv(results / "v5_identity_event_rates.csv",
             ("scope", "method", "event", "numerator_key", "denominator_key", "numerator", "denominator", "value", "unit"), summary["rates"])
    save_csv(results / "v5_identity_event_contrasts.csv",
             ("scope", "contrast", "stratum", "event", "opportunities", "corrections", "regressions", "net", "net_per_1000_opportunities"), summary["contrasts"])
    save_csv(results / "v5_identity_video_quarters.csv",
             ("scope", "scene", "method", "quarter", "event", "numerator", "denominator", "value", "unit"), summary["quarters"])
    public = {key: value for key, value in summary.items() if key != "quarters"}
    public["video_quarter_rate_rows"] = len(summary["quarters"])
    public["CSV_sha256"] = {name: sha256(results / name) for name in (
        "v5_identity_event_rates.csv", "v5_identity_event_contrasts.csv", "v5_identity_video_quarters.csv")}
    atomic_json(results / "v5_identity_analysis.json", public)
    target = docs / "V5_FULL_TEST_IDENTITY_ANALYSIS.md"
    temporary = target.with_name(target.name + ".partial")
    temporary.write_text(render_identity_analysis(summary, live))
    temporary.replace(target)
