#!/usr/bin/env python3


"""
Viseskalna rekurzivna memorija promena celih kombinacija.

- Svaki CSV se obradjuje nezavisno.
- Nema random postupaka.
- Nema filtera oblika kombinacije.
- Nema zaokruzivanja prosecne kombinacije.
- Tezine vremenskih skala uce se iz prethodnih ishoda.
- Konacni izbor ocenjuje zajednicke parove ili trojke.

CSV mora sadrzati sedam brojeva po redu, bez zaglavlja,
redosledom od najstarijeg do najnovijeg izvlacenja.
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np


DEFAULTS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4702_k82.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4702_k82_loto_2973.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4702_k82_loto_plus_1729.csv",
)

WINDOWS = (1, 4, 8, 16, 32)
HISTORY = 192
CALIBRATION = 48
VALIDATION = 48
TEST = 64
REFIT = 16

# Broj istorijskih analogija i red zajednickih odnosa.
CONFIGS = (
    (24, 2),
    (48, 2),
    (48, 3),
)

PAIR = np.array(
    list(itertools.combinations(range(7), 2))
)

TRIPLE = np.array(
    list(itertools.combinations(range(7), 3))
)


def read_csv(path):
    rows = []

    with open(
        path,
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for line, cells in enumerate(csv.reader(handle), 1):
            if not cells or all(
                not x.strip() for x in cells
            ):
                continue

            try:
                c = tuple(int(x.strip()) for x in cells)
            except ValueError as exc:
                raise ValueError(
                    f"{path}, red {line}: "
                    "ocekujem sedam celih brojeva"
                ) from exc

            if (
                len(c) != 7
                or len(set(c)) != 7
                or min(c) < 1
                or max(c) > 39
            ):
                raise ValueError(
                    f"{path}, red {line}: "
                    f"neispravna kombinacija {c}"
                )

            rows.append(tuple(sorted(c)))

    minimum = (
        max(WINDOWS)
        + HISTORY
        + CALIBRATION
        + VALIDATION
        + TEST
    )

    if len(rows) < minimum:
        raise ValueError(
            f"{path}: potrebno najmanje {minimum} redova"
        )

    return np.asarray(rows, dtype=np.int16)


def profile(rows):
    c = np.atleast_2d(rows).astype(float)

    distance = (
        c[:, PAIR[:, 1]]
        - c[:, PAIR[:, 0]]
    )

    relation_cdf = np.column_stack([
        (distance <= boundary).mean(axis=1)
        for boundary in (2, 5, 10, 20)
    ])

    # Opis pojedinacne cele kombinacije.
    # Nema istorijske tabele frekvencija pojedinacnih brojeva.
    return np.column_stack((
        c / 39,
        relation_cdf,
        (c[:, -1] - c[:, 0]) / 38,
    ))


def recursive_tree(sequence):
    """
    Rekurzivni opis vremenskog odsecka.

    Cuva opsti nivo i usmerene promene izmedju leve
    i desne polovine na svakom nivou stabla.
    """
    if len(sequence) == 1:
        return sequence[0], []

    middle = len(sequence) // 2

    left, left_details = recursive_tree(
        sequence[:middle]
    )

    right, right_details = recursive_tree(
        sequence[middle:]
    )

    detail = [
        (right - left)[None, :]
    ]

    detail += [
        np.concatenate((a, b), axis=0)
        for a, b in zip(
            left_details,
            right_details,
        )
    ]

    return (left + right) / 2, detail


def simplex(v):
    """Deterministicka projekcija tezina na nenegativnu sumu 1."""
    ordered = np.sort(v)[::-1]
    cumulative = np.cumsum(ordered) - 1
    indexes = np.arange(1, len(v) + 1)

    valid = np.flatnonzero(
        ordered - cumulative / indexes > 0
    )

    last = int(valid[-1])

    threshold = (
        cumulative[last] / (last + 1)
    )

    return np.maximum(v - threshold, 0)


def moment(matches, order):
    if order == 2:
        return matches * (matches - 1) / 42

    return (
        matches
        * (matches - 1)
        * (matches - 2)
        / 210
    )


class Memory:
    def __init__(self, rows):
        self.rows = rows

        self.inc = np.zeros((
            len(rows),
            39,
        ))

        self.inc[
            np.arange(len(rows))[:, None],
            rows - 1,
        ] = 1

        values = profile(rows)
        self.trees = []

        for window in WINDOWS:
            mean, detail = recursive_tree(
                values[:window]
            )

            sizes = (
                [1]
                + [len(level) for level in detail]
            )

            arrays = [
                np.zeros((
                    len(rows) + 1,
                    size,
                    values.shape[1],
                ))
                for size in sizes
            ]

            for t in range(window, len(rows) + 1):
                mean, detail = recursive_tree(
                    values[t - window:t]
                )

                arrays[0][t, 0] = mean

                for j, level in enumerate(detail, 1):
                    arrays[j][t] = level

            self.trees.append(arrays)

        self.distance_cache = {}
        self.calibration_cache = {}

    def distances(self, t):
        if t in self.distance_cache:
            return self.distance_cache[t]

        # Referentni ishodi uvek su zavrseni pre t.
        references = np.arange(
            max(max(WINDOWS), t - HISTORY),
            t,
        )

        if not len(references):
            raise ValueError(
                "Nema prethodnih obrazaca za poredjenje"
            )

        result = []

        for arrays in self.trees:
            distance = np.zeros(len(references))

            for layer in arrays:
                past = layer[references]

                pooled = past.reshape(
                    -1,
                    past.shape[-1],
                )

                scale = np.maximum(
                    np.quantile(pooled, 0.75, axis=0)
                    - np.quantile(pooled, 0.25, axis=0),
                    0.03,
                )

                difference = (
                    past - layer[t]
                ) / scale

                distance += np.mean(
                    difference * difference,
                    axis=(1, 2),
                )

            result.append(
                distance / len(arrays)
            )

        result = (
            references,
            np.asarray(result),
        )

        self.distance_cache[t] = result
        return result

    def neighbors(self, t, count):
        refs, distances = self.distances(t)

        weights = np.zeros_like(distances)

        for scale, distance in enumerate(distances):
            order = np.argsort(
                distance,
                kind="stable",
            )[:min(count, len(refs))]

            selected = distance[order]

            positive = selected[
                selected > 1e-12
            ]

            bandwidth = max(
                (
                    float(np.median(positive))
                    if len(positive)
                    else 1.0
                ),
                1e-9,
            )

            # Racionalno ponderisanje slicnosti.
            # Nema normalne raspodele.
            raw = 1 / (
                1 + selected / bandwidth
            ) ** 2

            weights[scale, order] = (
                raw / raw.sum()
            )

        return refs, weights

    def calibration_terms(self, t, config):
        key = (t, config)

        if key in self.calibration_cache:
            return self.calibration_cache[key]

        count, order = config

        refs, weights = self.neighbors(
            t,
            count,
        )

        incidence = self.inc[refs]

        gram = moment(
            incidence @ incidence.T,
            order,
        )

        actual = moment(
            incidence @ self.inc[t],
            order,
        )

        result = (
            weights @ gram @ weights.T,
            weights @ actual,
        )

        self.calibration_cache[key] = result
        return result


class Model:
    def __init__(
        self,
        memory,
        end,
        config,
        last_only=False,
    ):
        self.memory = memory
        self.config = config

        # Iskljucivo kontrolni postupak bez rekurzivnih nivoa.
        if last_only:
            self.scale_weights = np.array([
                1.0, 0.0, 0.0, 0.0, 0.0
            ])
            return

        first = max(
            max(WINDOWS) + 32,
            end - CALIBRATION,
        )

        if first >= end:
            raise ValueError(
                "Nedovoljno istorije za ucenje skala"
            )

        a = np.zeros((
            len(WINDOWS),
            len(WINDOWS),
        ))

        b = np.zeros(len(WINDOWS))

        # Svaki unutrasnji istorijski primer koristi samo
        # referentne ishode koji su prethodili tom primeru.
        for t in range(first, end):
            aa, bb = memory.calibration_terms(
                t,
                config,
            )

            a += aa
            b += bb

        a /= end - first
        b /= end - first

        # Pocetak od najbolje pojedinacne skale.
        weights = np.zeros(len(WINDOWS))

        weights[
            int(np.argmin(np.diag(a) - 2 * b))
        ] = 1

        lipschitz = max(
            float(np.linalg.eigvalsh(a)[-1]),
            1e-9,
        )

        # Ucenje zajednickih tezina vremenskih skala.
        for _ in range(500):
            updated = simplex(
                weights
                - (a @ weights - b) / lipschitz
            )

            if np.max(
                np.abs(updated - weights)
            ) < 1e-10:
                weights = updated
                break

            weights = updated

        self.scale_weights = weights

    def posterior(self, t):
        refs, weights = self.memory.neighbors(
            t,
            self.config[0],
        )

        posterior = (
            self.scale_weights @ weights
        )

        if (
            not np.all(np.isfinite(posterior))
            or not np.isclose(posterior.sum(), 1)
        ):
            raise RuntimeError(
                "Neispravne uslovne tezine"
            )

        return refs, posterior

    def predict(self, t):
        refs, posterior = self.posterior(t)
        incidence = self.memory.inc[refs]
        degree = self.config[1]

        if degree == 2:
            joint = (
                incidence.T
                @ (posterior[:, None] * incidence)
            )

            np.fill_diagonal(joint, 0)

            def score(candidates):
                c = (
                    np.asarray(candidates, dtype=int)
                    - 1
                )

                # Ocenjuju se svih 21 izabranih parova.
                # Ne sabiraju se svi partneri jednog broja.
                return joint[
                    c[:, PAIR[:, 0]],
                    c[:, PAIR[:, 1]],
                ].sum(axis=1) / 21

        else:
            joint = np.einsum(
                "r,ri,rj,rk->ijk",
                posterior,
                incidence,
                incidence,
                incidence,
                optimize=True,
            )

            def score(candidates):
                c = (
                    np.asarray(candidates, dtype=int)
                    - 1
                )

                return joint[
                    c[:, TRIPLE[:, 0]],
                    c[:, TRIPLE[:, 1]],
                    c[:, TRIPLE[:, 2]],
                ].sum(axis=1) / 35

        order = np.argsort(
            -posterior,
            kind="stable",
        )[:12]

        seeds = {
            tuple(map(
                int,
                self.memory.rows[refs[i]],
            ))
            for i in order
        }

        seeds.update(
            tuple(map(int, row))
            for row in self.memory.rows[
                max(0, t - 8):t
            ]
        )

        candidates = sorted(seeds)
        values = score(candidates)

        ranked = sorted(
            range(len(candidates)),
            key=lambda i: (
                -values[i],
                candidates[i],
            ),
        )

        states = [
            candidates[i]
            for i in ranked[:4]
        ]

        best = states[0]
        best_score = float(
            values[ranked[0]]
        )

        # Pretraga svih pojedinacnih zamena.
        # Ne namece oblik, grupe, parnost ili razmake.
        for _ in range(7):
            expanded = set(states)

            for c in states:
                for position in range(7):
                    rest = (
                        c[:position]
                        + c[position + 1:]
                    )

                    for number in range(1, 40):
                        if number not in rest:
                            expanded.add(
                                tuple(sorted(
                                    rest + (number,)
                                ))
                            )

            candidates = sorted(expanded)
            values = score(candidates)

            ranked = sorted(
                range(len(candidates)),
                key=lambda i: (
                    -values[i],
                    candidates[i],
                ),
            )

            next_states = [
                candidates[i]
                for i in ranked[:4]
            ]

            value = float(
                values[ranked[0]]
            )

            if value > best_score + 1e-12:
                best = next_states[0]
                best_score = value

            if next_states == states:
                break

            states = next_states

        assert len(best) == 7
        assert len(set(best)) == 7
        assert 1 <= best[0] < best[-1] <= 39

        return best


def hits(a, b):
    return len(set(a) & set(b))


def longest_run(row):
    current = longest = 1

    for a, b in zip(row, row[1:]):
        current = (
            current + 1
            if b == a + 1
            else 1
        )

        longest = max(
            longest,
            current,
        )

    return longest


def evaluate(
    memory,
    start,
    end,
    config,
    last_only=False,
):
    predictions = []
    model = None

    for t in range(start, end):
        if (
            model is None
            or (t - start) % REFIT == 0
        ):
            model = Model(
                memory,
                t,
                config,
                last_only,
            )

        predictions.append(
            model.predict(t)
        )

    scores = np.array([
        hits(
            prediction,
            memory.rows[t],
        )
        for t, prediction in zip(
            range(start, end),
            predictions,
        )
    ])

    return scores, predictions


def run(path):
    rows = read_csv(path)
    memory = Memory(rows)

    n = len(rows)

    test_start = n - TEST
    valid_start = n - TEST - VALIDATION

    print(
        f"\n{path}\n"
        f"Redova={n}; poslednja="
        f"{tuple(map(int, rows[-1]))}",
        flush=True,
    )

    validation = []

    for config in CONFIGS:
        scores, _ = evaluate(
            memory,
            valid_start,
            test_start,
            config,
        )

        validation.append(
            float(scores.mean())
        )

        print(
            f"Validacija {config}: "
            f"{scores.mean():.4f}",
            flush=True,
        )

    choice = max(
        range(len(CONFIGS)),
        key=lambda i: (
            validation[i],
            -i,
        ),
    )

    config = CONFIGS[choice]

    scores, predictions = evaluate(
        memory,
        test_start,
        n,
        config,
    )

    # Zasebno izabrana kontrola bez rekurzivnih nivoa.
    simple_validation = []

    for candidate in CONFIGS:
        base_scores, _ = evaluate(
            memory,
            valid_start,
            test_start,
            candidate,
            last_only=True,
        )

        simple_validation.append(
            float(base_scores.mean())
        )

    simple_choice = max(
        range(len(CONFIGS)),
        key=lambda i: (
            simple_validation[i],
            -i,
        ),
    )

    simple_config = CONFIGS[simple_choice]

    simple_scores, _ = evaluate(
        memory,
        test_start,
        n,
        simple_config,
        last_only=True,
    )

    print(
        "Kontrola bez rekurzivnih nivoa: "
        f"{simple_scores.mean():.4f}; "
        f"parametri={simple_config}"
    )

    control = np.array([
        hits(
            rows[t - 1],
            rows[t],
        )
        for t in range(test_start, n)
    ])

    print(
        "Zamrznuti parametri: "
        f"suseda={config[0]}, "
        f"zajednicki odnosi reda={config[1]}"
    )

    print(
        f"Test {TEST}: model={scores.mean():.4f}; "
        "Kontrola: prethodno izvuceno kolo="
        f"{control.mean():.4f}"
    )

    print(
        "Distribucija pogodaka 0..7:",
        np.bincount(
            scores,
            minlength=8,
        ).tolist(),
    )

    print(
        f"3+={int((scores >= 3).sum())}/{TEST}; "
        f"4+={int((scores >= 4).sum())}/{TEST}"
    )

    replays = sum(
        any(
            tuple(row) == prediction
            for row in rows[:t]
        )
        for t, prediction in zip(
            range(test_start, n),
            predictions,
        )
    )

    long_runs = sum(
        longest_run(prediction) >= 5
        for prediction in predictions
    )

    print(
        f"Razlicitih predikcija="
        f"{len(set(predictions))}/{TEST}; "
        f"istorijskih ponavljanja="
        f"{replays}/{TEST}; "
        f"nizova 5+={long_runs}"
    )

    model = Model(
        memory,
        n,
        config,
    )

    result = model.predict(n)

    assert result == model.predict(n)

    print(
        "Naucene tezine skala:",
        dict(zip(
            WINDOWS,
            np.round(
                model.scale_weights,
                6,
            ).tolist(),
        )),
    )

    print(
        f"NEXT red {n + 1}: "
        + ",".join(map(str, result))
    )

    print(
        "NEXT postoji u istoriji:",
        any(
            tuple(row) == result
            for row in rows
        ),
    )

    print(
        "NEXT najduzi niz:",
        longest_run(result),
    )

    print(
        "NEXT brojeva <=19:",
        sum(
            number <= 19
            for number in result
        ),
    )

    if scores.mean() <= control.mean():
        print(
            "Na ovom testu model nije imao "
            "bolji prosek od kontrole."
        )

    return result


def self_check():
    sequence = np.arange(
        4,
        dtype=float,
    ).reshape(4, 1)

    mean, levels = recursive_tree(sequence)

    assert np.allclose(mean, [1.5])
    assert np.allclose(levels[0], [[2.0]])
    assert np.allclose(
        levels[1],
        [[1.0], [1.0]],
    )

    assert np.allclose(
        simplex(np.array([2.0, 0.0, -1.0])),
        [1.0, 0.0, 0.0],
    )

    rows = np.array([
        sorted({
            (3 * i + 5 * j) % 39 + 1
            for j in range(7)
        })
        for i in range(140)
    ], dtype=np.int16)

    changed = rows.copy()
    changed[120:] = rows[0]

    a = Memory(rows)
    b = Memory(changed)

    # Buduci ishodi ne menjaju ranije opise ili distance.
    for t in range(118, 121):
        ar, ad = a.distances(t)
        br, bd = b.distances(t)

        assert np.array_equal(ar, br)
        assert np.array_equal(ad, bd)

    # Provera i za parove i za trojke.
    for config in (
        CONFIGS[0],
        CONFIGS[-1],
    ):
        first = Model(a, 120, config)
        second = Model(b, 120, config)

        assert np.array_equal(
            first.scale_weights,
            second.scale_weights,
        )

        assert (
            first.predict(120)
            == second.predict(120)
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv",
        nargs="*",
        default=list(DEFAULTS),
    )

    args = parser.parse_args()

    self_check()

    print(
        "Fractal Recursion V1: "
        "rekurzivna memorija, "
        "bez random i filtera oblika."
    )

    results = []

    for path in args.csv:
        results.append((
            path,
            run(path),
        ))

    # Jednake izlaze prijavljuje, ne menja ih vestacki.
    for i, (left, a) in enumerate(results):
        for right, b in results[i + 1:]:
            if a == b:
                print(
                    f"ISTA NEXT: "
                    f"{Path(left).name} / "
                    f"{Path(right).name}: {a}"
                )


if __name__ == "__main__":
    main()



"""
/Users/4c/Desktop/GHQ/data/loto7_4702_k82.csv
Redova=4702; poslednja=(8, 10, 13, 21, 22, 23, 32)
Validacija (24, 2): 1.3750
Validacija (48, 2): 1.2917
Validacija (48, 3): 1.3333
Kontrola bez rekurzivnih nivoa: 1.4219; parametri=(48, 3)
Zamrznuti parametri: suseda=24, zajednicki odnosi reda=2
Test 64: model=1.4219; Kontrola: prethodno izvuceno kolo=1.1719
Distribucija pogodaka 0..7: [12, 22, 22, 7, 1, 0, 0, 0]
3+=8/64; 4+=1/64
Razlicitih predikcija=60/64; istorijskih ponavljanja=28/64; nizova 5+=0
Naucene tezine skala: {1: 0.271366, 4: 0.195882, 8: 0.108914, 16: 0.109727, 32: 0.31411}
NEXT red 4703: 4,9,14,16,19,24,32
NEXT postoji u istoriji: True
NEXT najduzi niz: 1
NEXT brojeva <=19: 5



/Users/4c/Desktop/GHQ/data/loto7_4702_k82_loto_2973.csv
Redova=2973; poslednja=(1, 2, 8, 26, 29, 35, 39)
Validacija (24, 2): 1.1875
Validacija (48, 2): 1.1250
Validacija (48, 3): 1.2083
Kontrola bez rekurzivnih nivoa: 1.2031; parametri=(48, 2)
Zamrznuti parametri: suseda=48, zajednicki odnosi reda=3
Test 64: model=1.3750; Kontrola: prethodno izvuceno kolo=1.3281
Distribucija pogodaka 0..7: [16, 18, 21, 8, 1, 0, 0, 0]
3+=9/64; 4+=1/64
Razlicitih predikcija=49/64; istorijskih ponavljanja=64/64; nizova 5+=0
Naucene tezine skala: {1: 0.234481, 4: 0.188346, 8: 0.123557, 16: 0.17496, 32: 0.278655}
NEXT red 2974: 1,8,18,19,30,33,39
NEXT postoji u istoriji: True
NEXT najduzi niz: 2
NEXT brojeva <=19: 4



/Users/4c/Desktop/GHQ/data/loto7_4702_k82_loto_plus_1729.csv
Redova=1729; poslednja=(8, 10, 13, 21, 22, 23, 32)
Validacija (24, 2): 1.2500
Validacija (48, 2): 0.9792
Validacija (48, 3): 1.1458
Kontrola bez rekurzivnih nivoa: 1.4062; parametri=(48, 3)
Zamrznuti parametri: suseda=24, zajednicki odnosi reda=2
Test 64: model=1.3594; Kontrola: prethodno izvuceno kolo=1.2656
Distribucija pogodaka 0..7: [13, 24, 18, 9, 0, 0, 0, 0]
3+=9/64; 4+=0/64
Razlicitih predikcija=58/64; istorijskih ponavljanja=24/64; nizova 5+=0
Naucene tezine skala: {1: 0.244552, 4: 0.242704, 8: 0.130619, 16: 0.181702, 32: 0.200422}
NEXT red 1730: 1,3,7,17,24,25,32
NEXT postoji u istoriji: True
NEXT najduzi niz: 2
NEXT brojeva <=19: 4
"""



"""
Rekurzivno prepoznavanje sličnih obrazaca na nekoliko vremenskih skala. 
Ispitivanje da li sličan tok promena distribucije vodi ka sličnom narednom ishodu.

1. Opisujem cele kombinacije i njihovu distribuciju kroz kraće i duže delove istorije.
2. Rekurzivno delim vremenske odsečke na manje delove. Poredim njihov opšti oblik i redosled promena unutar njih.
3. Učim značaj pojedinačnih skala: koje sličnosti pomažu ocenjivanju narednih kombinacija, a koje ne doprinose.
4. Deterministički ocenjujem cele kombinacije kandidata, bez nametanja jednakih razmaka, grupa ili zabrane uzastopnih brojeva.

Sličan treba da bude tok promena distribucije, a ne samo poslednja kombinacija. 
Konstrukcija inspirisana višeskalnom analizom i nelinearnim predviđanjem. 
Literatura prikazuje primene takvih metoda na vremenske serije; sam fraktalni izgled ne dokazuje prediktivnost. Sugihara i May, 1990, višeskalna analiza rekurencije.
Ovde vidim smislen pravac za proveru: višeskalno pamćenje promena celih kombinacija.


Fraktalnu rekurziju vredi razmotriti kao višeskalno prepoznavanje obrazaca: porediti kako se distribucija celih kombinacija menja kroz kraće i duže delove istorije, pa učiti koji takvi tokovi promena pomažu predikciji naredne kombinacije.
Rekurzija bi značila da veći vremenski obrazac razlažem na manje i proveravamo njihove odnose. Model bi sam učio značaj tih odnosa, bez nametanja nizova, grupa ili ravnomernih razmaka.


Rekurzivno poredi tokove promena na skalama 1, 4, 8, 16 i 32 kola. 
Značaj skala uči iz prethodnih ishoda, a celu kombinaciju ocenjuje preko zajedničkih parova ili trojki.

Poslednja 64 kola:

CSV	        Fractal V1	Bez rekurzivnih nivoa	Prethodno kolo
4702	    1,4219	    1,4219	                1,1719
2973	    1,3750	    1,2031	                1,3281
1729	    1,3594	    1,4062	                1,2656

U testovima nije proizveo nizove od pet ili više uzastopnih brojeva. 
Ipak, ponavlja istorijske kombinacije: 28/64, 64/64 i 24/64; sva tri sadašnja NEXT izlaza postoje u istoriji. 
Rekurzija je poboljšala prosek na Loto fajlu, izjednačila kontrolu na zbirnom i bila slabija na Plus fajlu.
"""
