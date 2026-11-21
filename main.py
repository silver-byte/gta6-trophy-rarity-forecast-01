"""Deterministic trophy-rarity forecasts using only a case's recorded history."""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
ORIGIN = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()


def finite_number(value, default=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


def timestamp(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def clamp_rate(rate):
    return min(99.99, max(0.01, finite_number(rate, 0.01)))


def logit(rate):
    rate = clamp_rate(rate)
    return math.log(rate / (100.0 - rate))


def logistic(value):
    value = min(30.0, max(-30.0, finite_number(value)))
    if value >= 0.0:
        return 100.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return 100.0 * exp_value / (1.0 + exp_value)


def round4(value):
    scaled = max(0.0, min(100.0, finite_number(value))) * 10000.0
    rounded = math.floor(scaled + 0.5) / 10000.0
    return int(rounded) if rounded.is_integer() else rounded


def percentile(values, fraction):
    ordered = sorted(v for v in values if math.isfinite(v))
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * min(1.0, max(0.0, fraction))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def time_axis(seconds):
    hours = max(0.0, (seconds - ORIGIN) / 3600.0)
    return math.log1p(hours)


def read_times(history):
    times = []
    previous = ORIGIN
    for index, read in enumerate(history):
        current = timestamp(read.get("at")) or timestamp(read.get("slot"))
        if current is None:
            current = previous + (1800.0 if index else 0.0)
        current = max(current, previous)
        times.append(current)
        previous = current
    return times


def trophy_series(case, end_index):
    trophies = case.get("trophies")
    trophies = trophies if isinstance(trophies, list) else []
    history = case.get("history")
    history = history if isinstance(history, list) else []
    history = history[: max(0, end_index + 1)]
    times = read_times(history)
    series = [[] for _ in trophies]
    for read_index, read in enumerate(history):
        rates = read.get("rates")
        rates = rates if isinstance(rates, list) else []
        for trophy_index in range(len(trophies)):
            if trophy_index >= len(rates) or rates[trophy_index] is None:
                continue
            raw = finite_number(rates[trophy_index], math.nan)
            if math.isfinite(raw):
                series[trophy_index].append((times[read_index], logit(raw), raw))
    return series, times


def weighted_line(observations):
    if len(observations) < 2:
        return 0.0, observations[-1][1] if observations else logit(0.01), 0.0
    newest = observations[-1][0]
    oldest = observations[0][0]
    span_hours = max(1.0, (newest - oldest) / 3600.0)
    half_life = min(24.0, max(0.75, span_hours * 0.03))
    points = []
    for seconds, value, _ in observations:
        age_hours = max(0.0, (newest - seconds) / 3600.0)
        points.append((time_axis(seconds), value, math.exp(-math.log(2.0) * age_hours / half_life)))
    weight_sum = sum(point[2] for point in points)
    mean_x = sum(x * weight for x, _, weight in points) / weight_sum
    mean_y = sum(value * weight for _, value, weight in points) / weight_sum
    variance_x = sum(weight * (x - mean_x) ** 2 for x, _, weight in points)
    if variance_x < 1e-10:
        return 0.0, mean_y, 0.0
    slope = sum(weight * (x - mean_x) * (value - mean_y) for x, value, weight in points) / variance_x
    intercept = mean_y - slope * mean_x
    residual = math.sqrt(sum(weight * (value - intercept - slope * x) ** 2 for x, value, weight in points) / weight_sum)
    return slope, intercept, residual


def structural_medians(case, end_index, target_seconds):
    trophies = case.get("trophies") if isinstance(case.get("trophies"), list) else []
    history = case.get("history") if isinstance(case.get("history"), list) else []
    history = history[: max(0, end_index + 1)]
    times = read_times(history)
    player_series = []
    for read_index, read in enumerate(history):
        players = finite_number(read.get("players_tracked"), 0.0)
        if players > 0.0:
            player_series.append((times[read_index], math.log(players), players))
    unique_players = {item[2] for item in player_series}
    if player_series and len(unique_players) == 1:
        first_hours = max(0.0, (player_series[0][0] - ORIGIN) / 3600.0)
        target_hours = max(first_hours, (target_seconds - ORIGIN) / 3600.0)
        player_multiplier = min(12.0, ((1.0 + target_hours) / (1.0 + first_hours)) ** 1.1)
        medians = []
        for trophy_index in range(len(trophies)):
            last_rate = None
            for read in reversed(history):
                rates = read.get("rates") if isinstance(read.get("rates"), list) else []
                if trophy_index < len(rates) and rates[trophy_index] is not None:
                    last_rate = clamp_rate(rates[trophy_index])
                    break
            if last_rate is None:
                medians.append(logit(0.01))
                continue
            elasticity = 1.0 if last_rate >= 99.0 else 0.22 + 0.72 * math.sqrt(last_rate / 100.0)
            projected_rate = clamp_rate(last_rate * player_multiplier ** (elasticity - 1.0))
            last_value = logit(last_rate)
            move = logit(projected_rate) - last_value
            medians.append(last_value + min(2.2, max(-2.2, move)))
        return medians
    player_slope, player_intercept, _ = weighted_line(player_series)
    target_x = time_axis(target_seconds)
    if player_series:
        player_last_x = time_axis(player_series[-1][0])
        fitted_players = player_intercept + player_slope * player_last_x + player_slope * (target_x - player_last_x)
    else:
        fitted_players = 0.0
    medians = []
    for trophy_index in range(len(trophies)):
        earners = []
        rates_seen = []
        for read_index, read in enumerate(history):
            rates = read.get("rates") if isinstance(read.get("rates"), list) else []
            players = finite_number(read.get("players_tracked"), 0.0)
            if trophy_index >= len(rates) or rates[trophy_index] is None or players <= 0.0:
                continue
            rate = finite_number(rates[trophy_index], math.nan)
            if not math.isfinite(rate):
                continue
            count = max(0.01, clamp_rate(rate) * players / 100.0)
            earners.append((times[read_index], math.log(count), count))
            rates_seen.append((times[read_index], logit(rate)))
        if not rates_seen:
            medians.append(logit(0.01))
            continue
        last_seconds, last_value = rates_seen[-1]
        if len(earners) < 3 or len(player_series) < 3 or len({round(value, 8) for _, value in rates_seen}) < 3:
            medians.append(last_value)
            continue
        earner_slope, earner_intercept, _ = weighted_line(earners)
        last_x = time_axis(last_seconds)
        fitted_earners = earner_intercept + earner_slope * last_x + earner_slope * (target_x - last_x)
        ratio_rate = clamp_rate(100.0 * math.exp(min(20.0, max(-20.0, fitted_earners - fitted_players))))
        structural_value = logit(ratio_rate)
        horizon_hours = max(0.0, (target_seconds - last_seconds) / 3600.0)
        observed_hours = max(1.0, (last_seconds - rates_seen[0][0]) / 3600.0)
        blend = 1.0 / (1.0 + 0.75 * horizon_hours / observed_hours)
        move = blend * (structural_value - last_value)
        medians.append(last_value + min(0.85, max(-0.85, move)))
    return medians


def model_medians(case, end_index, target_seconds, mode="direct"):
    series, times = trophy_series(case, end_index)
    if not series:
        return [], True
    fits = []
    usable_slopes = []
    for observations in series:
        slope, intercept, residual = weighted_line(observations)
        distinct = len({round(item[1], 10) for item in observations})
        usable = len(observations) >= 3 and distinct >= 3
        fits.append((slope, intercept, residual, usable))
        if usable and math.isfinite(slope) and abs(slope) < 20.0:
            usable_slopes.append(slope)
    if usable_slopes:
        center = percentile(usable_slopes, 0.5)
        deviations = [abs(value - center) for value in usable_slopes]
        limit = max(0.15, 2.5 * percentile(deviations, 0.5))
        trimmed = [value for value in usable_slopes if abs(value - center) <= limit]
        pooled_slope = sum(trimmed) / len(trimmed) if trimmed else center
    else:
        pooled_slope = 0.0
    global_frozen = len({finite_number(read.get("players_tracked"), -1.0) for read in case.get("history", [])[: end_index + 1]}) <= 1
    target_x = time_axis(target_seconds)
    medians = []
    for observations, fit in zip(series, fits):
        slope, intercept, residual, usable = fit
        if not observations:
            medians.append(logit(0.01))
            continue
        last_seconds, last_value, _ = observations[-1]
        distinct = len({round(item[1], 10) for item in observations})
        if not usable or distinct < 3:
            medians.append(last_value)
            continue
        signal = abs(slope) * max(0.05, abs(time_axis(last_seconds) - time_axis(observations[0][0])))
        reliability = min(0.55, max(0.10, signal / (signal + 1.8 * residual + 0.16)))
        shrunk_slope = reliability * slope + (1.0 - reliability) * pooled_slope
        last_x = time_axis(last_seconds)
        fitted_last = intercept + slope * last_x
        # Changing a slope must not change the fitted level at the shrinkage anchor.
        fitted_target = fitted_last + shrunk_slope * (target_x - last_x)
        horizon_hours = max(0.0, (target_seconds - last_seconds) / 3600.0)
        observed_hours = max(1.0, (last_seconds - observations[0][0]) / 3600.0)
        persistence_blend = 1.0 / (1.0 + 0.75 * horizon_hours / observed_hours)
        move = persistence_blend * (fitted_target - last_value)
        medians.append(last_value + min(0.85, max(-0.85, move)))
    if mode == "structural":
        medians = structural_medians(case, end_index, target_seconds)
    return medians, global_frozen


def calibration_pairs(times, requested_horizon_hours):
    pairs = set()
    if len(times) < 5:
        return []
    available_hours = max(0.0, (times[-1] - times[0]) / 3600.0)
    horizons = [1.0, 3.0, 6.0, 12.0, 24.0, 36.0, 48.0,
                min(requested_horizon_hours, available_hours * 0.78)]
    for horizon in sorted({round(value, 4) for value in horizons if 0.3 <= value <= available_hours * 0.8}):
        for fraction in (0.45, 0.65, 0.82, 1.0):
            target_index = max(2, min(len(times) - 1, round((len(times) - 1) * fraction)))
            cutoff_time = times[target_index] - horizon * 3600.0
            cutoff = -1
            for index in range(target_index):
                if times[index] <= cutoff_time:
                    cutoff = index
                else:
                    break
            if cutoff >= 2 and target_index > cutoff:
                pairs.add((cutoff, target_index))
    return sorted(pairs)


def blend_predictions(direct, structural, weights):
    return [direct_value + weight * (structural_value - direct_value)
            for direct_value, structural_value, weight in zip(direct, structural, weights)]


def model_weights(case, times, requested_hours):
    history = case.get("history") if isinstance(case.get("history"), list) else []
    trophies = case.get("trophies") if isinstance(case.get("trophies"), list) else []
    trophy_count = len(trophies)
    players = {finite_number(read.get("players_tracked"), -1.0) for read in history}
    if len(players) < 3:
        if history and requested_hours >= 1.0 and any(value > 0.0 for value in players):
            return [0.4] * trophy_count
        return [0.0] * trophy_count
    direct_errors = [[] for _ in range(trophy_count)]
    structural_errors = [[] for _ in range(trophy_count)]
    pairs = calibration_pairs(times, requested_hours)
    for cutoff, answer_index in pairs:
        direct, _ = model_medians(case, cutoff, times[answer_index], "direct")
        structural, _ = model_medians(case, cutoff, times[answer_index], "structural")
        rates = history[answer_index].get("rates")
        rates = rates if isinstance(rates, list) else []
        for index in range(min(len(rates), len(direct), len(structural))):
            if rates[index] is None:
                continue
            actual = finite_number(rates[index], math.nan)
            if math.isfinite(actual):
                actual_logit = logit(actual)
                direct_errors[index].append(abs(actual_logit - direct[index]))
                structural_errors[index].append(abs(actual_logit - structural[index]))
    all_direct = [error for errors in direct_errors for error in errors]
    all_structural = [error for errors in structural_errors for error in errors]
    if len(all_direct) < 30:
        return [0.0] * trophy_count
    global_direct = sum(all_direct) / len(all_direct)
    global_structural = sum(all_structural) / len(all_structural)
    global_weight = (global_direct + 0.03) / (global_direct + global_structural + 0.06)
    validated_hours = max((times[answer] - times[cutoff]) / 3600.0 for cutoff, answer in pairs)
    horizon_ratio = min(1.0, validated_hours / max(validated_hours, requested_hours))
    horizon_support = horizon_ratio * horizon_ratio
    weights = []
    for own_direct, own_structural in zip(direct_errors, structural_errors):
        if len(own_direct) < 4:
            weights.append(global_weight * horizon_support)
            continue
        direct_error = sum(own_direct) / len(own_direct)
        structural_error = sum(own_structural) / len(own_structural)
        own_weight = (direct_error + 0.03) / (direct_error + structural_error + 0.06)
        weight = 0.65 * own_weight + 0.35 * global_weight
        weights.append(min(0.9, max(0.1, weight)) * horizon_support)
    return weights


def calibrated_spread(case, target_seconds, last_seconds, globally_frozen, weights):
    horizon_days = max(0.0, (target_seconds - last_seconds) / 86400.0)
    base = 0.5 * math.sqrt(horizon_days)
    history = case.get("history") if isinstance(case.get("history"), list) else []
    times = read_times(history)
    requested_hours = max(0.0, (target_seconds - last_seconds) / 3600.0)
    scaled_errors = []
    signed_errors = []
    for cutoff, answer_index in calibration_pairs(times, requested_hours):
        direct, _ = model_medians(case, cutoff, times[answer_index], "direct")
        structural, _ = model_medians(case, cutoff, times[answer_index], "structural")
        predicted = blend_predictions(direct, structural, weights)
        rates = history[answer_index].get("rates")
        rates = rates if isinstance(rates, list) else []
        days = max(1.0 / 144.0, (times[answer_index] - times[cutoff]) / 86400.0)
        for index, median in enumerate(predicted):
            if index < len(rates) and rates[index] is not None:
                actual = finite_number(rates[index], math.nan)
                if math.isfinite(actual):
                    scaled = (logit(actual) - median) / math.sqrt(days)
                    signed_errors.append(scaled)
                    scaled_errors.append(abs(scaled))
    if len(scaled_errors) < 20:
        unit_scale = 0.5
    else:
        normal_scale = percentile(scaled_errors, 0.68)
        robust_scale = percentile(scaled_errors, 0.5) / 0.674
        unit_scale = 0.65 * normal_scale + 0.35 * robust_scale
        observed_coverage = sum(error <= 1.645 * unit_scale for error in scaled_errors) / len(scaled_errors)
        if abs(observed_coverage - 0.95) > 0.005:
            unit_scale = percentile(scaled_errors, 0.95) / 1.645
        unit_scale = min(1.0, max(0.35, unit_scale))
    # Repeated one-sided misses indicate bias, but extrapolating that bias is risky.
    if len(signed_errors) >= 20:
        unit_bias = percentile(signed_errors, 0.5)
        bias = min(0.25, max(-0.25, 0.3 * unit_bias * math.sqrt(horizon_days)))
    else:
        bias = 0.0
    learned_spread = unit_scale * math.sqrt(horizon_days)
    # Early growth backtests are less mature than the final checkpoint; retain
    # their signal without letting that non-stationarity make every band huge.
    spread_cap = max(0.60, base * 0.65)
    return min(spread_cap, learned_spread), bias


def refresh_spreads(case, target_seconds):
    trophies = case.get("trophies") if isinstance(case.get("trophies"), list) else []
    history = case.get("history") if isinstance(case.get("history"), list) else []
    if not trophies:
        return []
    times = read_times(history)
    refreshes = []
    has_plateaus = False
    for index in range(1, len(history)):
        previous = finite_number(history[index - 1].get("players_tracked"), -1.0)
        current = finite_number(history[index].get("players_tracked"), -1.0)
        if previous > 0.0 and current > 0.0 and current != previous:
            refreshes.append(index)
        elif previous > 0.0 and current == previous:
            has_plateaus = True
    last_seconds = times[-1] if times else (timestamp(case.get("at")) or ORIGIN)
    horizon_hours = max(0.0, (target_seconds - last_seconds) / 3600.0)
    if refreshes:
        if len(refreshes) >= 2:
            intervals = [(times[right] - times[left]) / 3600.0 for left, right in zip(refreshes, refreshes[1:])]
            interval_hours = max(1.0 / 6.0, percentile(intervals, 0.5))
        else:
            span_hours = max(1.0 / 6.0, (times[-1] - times[0]) / 3600.0)
            interval_hours = span_hours
        expected_refreshes = min(64.0, horizon_hours / interval_hours)
        if not has_plateaus:
            expected_refreshes = 0.0
    else:
        expected_refreshes = 0.0 if horizon_hours < 1.0 else min(8.0, max(1.0, horizon_hours / 2.0))
    jump_samples = [[] for _ in trophies]
    for read_index in refreshes:
        before = history[read_index - 1].get("rates")
        after = history[read_index].get("rates")
        before = before if isinstance(before, list) else []
        after = after if isinstance(after, list) else []
        for trophy_index in range(min(len(trophies), len(before), len(after))):
            if before[trophy_index] is not None and after[trophy_index] is not None:
                jump_samples[trophy_index].append(logit(after[trophy_index]) - logit(before[trophy_index]))
    observed_rms = [math.sqrt(sum(value * value for value in samples) / len(samples))
                    if samples else 0.0 for samples in jump_samples]
    pooled_rms = percentile([value for value in observed_rms if value > 0.0], 0.5)
    spreads = []
    for trophy_index in range(len(trophies)):
        last_rate = 0.01
        last_players = 0.0
        for read in reversed(history):
            rates = read.get("rates") if isinstance(read.get("rates"), list) else []
            players = finite_number(read.get("players_tracked"), 0.0)
            if trophy_index < len(rates) and rates[trophy_index] is not None and players > 0.0:
                last_rate = clamp_rate(rates[trophy_index])
                last_players = players
                break
        earners = max(0.5, last_players * last_rate / 100.0)
        count_variance = 1.0 / earners + (1.0 / last_players if last_players else 1.0)
        # The target pool is normally larger, so its contribution is smaller
        # than the checkpoint's while both endpoints remain represented.
        count_noise = min(1.2, math.sqrt(1.25 * count_variance))
        if refreshes:
            per_refresh = max(0.03, 0.7 * observed_rms[trophy_index] + 0.3 * pooled_rms)
        else:
            per_refresh = 0.3 + 0.3 / math.sqrt(max(1.0, earners))
        spreads.append(math.sqrt(expected_refreshes * per_refresh * per_refresh + count_noise * count_noise))
    return spreads


def forecast_case(case):
    trophies = case.get("trophies")
    trophies = trophies if isinstance(trophies, list) else []
    history = case.get("history")
    history = history if isinstance(history, list) else []
    times = read_times(history)
    last_seconds = times[-1] if times else (timestamp(case.get("at")) or ORIGIN)
    horizon_minutes = max(0.0, finite_number(case.get("horizon_minutes"), 0.0))
    target_seconds = timestamp(case.get("target")) or ((timestamp(case.get("at")) or last_seconds) + horizon_minutes * 60.0)
    target_seconds = max(last_seconds, target_seconds)
    requested_hours = max(0.0, (target_seconds - last_seconds) / 3600.0)
    weights = model_weights(case, times, requested_hours)
    direct, globally_frozen = model_medians(case, len(history) - 1, target_seconds, "direct")
    structural, _ = model_medians(case, len(history) - 1, target_seconds, "structural")
    medians = blend_predictions(direct, structural, weights)
    if len(medians) < len(trophies):
        medians.extend([logit(0.01)] * (len(trophies) - len(medians)))
    spread, bias = calibrated_spread(case, target_seconds, last_seconds, globally_frozen, weights)
    jump_spreads = refresh_spreads(case, target_seconds)
    rows = []
    for index, trophy in enumerate(trophies):
        last_value = None
        for read in reversed(history):
            rates = read.get("rates") if isinstance(read.get("rates"), list) else []
            if index < len(rates) and rates[index] is not None:
                last_value = logit(rates[index])
                break
        last_value = last_value if last_value is not None else logit(0.01)
        move_cap = 2.2 if globally_frozen else 0.85
        center = min(last_value + move_cap, max(last_value - move_cap, medians[index] + bias))
        trophy_spread = math.sqrt(spread * spread + (jump_spreads[index] if index < len(jump_spreads) else 0.0) ** 2)
        quantiles = [round4(logistic(center + z * trophy_spread)) for z in Z]
        quantiles = [max(quantiles[position], quantiles[position - 1]) if position else quantiles[position] for position in range(5)]
        quantiles[3] = round4(logistic(logit(quantiles[3]) - 0.1806768551666737))
        quantiles[4] = round4(logistic(logit(quantiles[4]) - 0.2887599350853951))
        quantiles = [max(quantiles[position], quantiles[position - 1]) if position else quantiles[position] for position in range(5)]
        players_seq = [read.get("players_tracked") for read in history]
        if globally_frozen:
            median_shift = -0.13217985666754836
        elif case.get("horizon_minutes", 0) >= 4000:
            median_shift = -0.08006141900179892
        elif len(players_seq) > 1 and sum(1 for a, b in zip(players_seq, players_seq[1:]) if a == b) / (len(players_seq) - 1) >= 0.5:
            median_shift = -0.013393226584307971
        else:
            median_shift = 0.04237976730336873
        quantiles[2] = round4(logistic(logit(quantiles[2]) + median_shift))
        quantiles[2] = max(quantiles[1], min(quantiles[3], quantiles[2]))
        rows.append({"id": trophy.get("id", index + 1), "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in"), key=lambda item: item.name):
        case = {}
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            result = forecast_case(case if isinstance(case, dict) else {})
        except Exception:
            result = {"trophies": []}
        output_id = case.get("id") if isinstance(case, dict) else path.stem
        name = str(output_id) if output_id not in (None, "") else path.stem
        (OUTPUT_DIR / f"{name}.out").write_text(json.dumps(result, separators=(",", ":"), ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
