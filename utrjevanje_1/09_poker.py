# =============================================================================
# Poker
#
# Pri podnalogah, kjer je prisotna (psevdo)naključnost, pravilnosti rešitev
# ni mogoče povsem preveriti, zato bo tam Tomo morda zadovoljen tudi z
# rešitvami, ki niso povsem pravilne. Seveda pa lahko svoje rešitve primerjate
# z uradnimi.
# =====================================================================@024228=
# 1. podnaloga
# Definirajte funkcijo `nov_kup()`, ki naj vrne seznam, ki
# predstavlja klasičen kup kart.
# Vsaka karta naj bo predstavljena kot par `(višina, barva)`. Tako
# na primer `(12, "pik")` predstavlja pikovo damo,
# `(10, "križ")` pa križevo desetko.
# =============================================================================

# =====================================================================@024229=
# 2. podnaloga
# Sestavite funkcijo `premesaj(karte)`, ki seznam kart čim bolj naključno
# premeša, ne vrne pa ničesar.
# 
# Pomagate si lahko s funkcijo `shuffle` iz modula `random`.
# =============================================================================

# =====================================================================@024230=
# 3. podnaloga
# Predpostavimo, da igro *Poker* igra $n$ igralcev. Pri igri najprej karte
# premešamo, nato pa vsakemu od igralcev podelimo dve karti. Sestavite funkcijo
# `razdeli_karte(igralci, karte)`, ki sprejme karte in seznam imen igralcev,
# vrne pa slovar, katerega ključi so imena igralcev, vrednost pri vsakem od njih
# pa je seznam, ki vsebuje natanko dve karti.
# 
#     >>> karte = nov_kup()
#     >>> premesaj(karte)
#     >>> razdeli_karte(["Ana", "Bine", "Cene"], karte)
#     {'Cene': [(13, 'srce'), (5, 'križ')], 'Bine': [(8, 'kara'), (3, 'kara')], 'Ana': [(9, 'srce'), (6, 'križ')]}
# =============================================================================

# =====================================================================@024231=
# 4. podnaloga
# Sestavite funkcijo `odpri_skupne_karte(karte)`, ki s seznama kart odstrani
# vrhnjih pet kart in jih vrne kot seznam.
# =============================================================================

# =====================================================================@024232=
# 5. podnaloga
# Sestavite funkcijo `na_dva_dela(karte)`, ki sprejme seznam kart in vrne
# dva seznama: prvi je seznam vseh številk, ki se pojavijo na danih kartah,
# drugi pa seznam vseh barv, ki se pojavijo.
# 
#     >>> na_dva_dela([(10, 'križ'), (12, 'srce'), (12, 'pik'), (10, 'kara'), (12, 'križ')])
#     ([10, 12, 12, 10, 12], ['križ', 'srce', 'pik', 'kara', 'križ'])
# =============================================================================

# =====================================================================@024233=
# 6. podnaloga
# Sestavite funkcijo `tvorijo_lestvico(karte)`, ki sprejme seznam kart in vrne
# `True`, če in samo če številke v kartah tvorijo lestvico.
# 
#     >>> tvorijo_lestvico([(10, 'križ'), (12, 'srce')])
#     False
#     >>> tvorijo_lestvico([(10, 'križ'), (12, 'srce'), (11, 'križ')])
#     True
# =============================================================================

# =====================================================================@024234=
# 7. podnaloga
# Sestavite funkcijo `kolikokrat_se_pojavi_katera_stevilka(karte)`, ki sprejme seznam kart in vrne
# slovar. Ključi v tem slovarju so številke, ki se pojavijo na kartah,
# vrednosti pa števila pojavitev vsake od teh številk.
# 
#     >>> kolikokrat_se_pojavi_katera_stevilka([(10, 'križ'), (12, 'srce'), (12, 'pik'), (10, 'kara'), (12, 'križ')])
#     {10: 2, 12: 3}
# =============================================================================

# =====================================================================@024235=
# 8. podnaloga
# Sestavite funkcijo `vrednost(peterka)`, ki sprejme seznam petih kart in vrne
# *kvaliteto kart* v skladu z naslednjo ocenjevalno shemo:
# 
#     9 Barvna lestvica
#     8 Poker
#     7 Full house
#     6 Barve
#     5 Lestvica
#     4 Tris
#     3 Dva para
#     2 En par
#     1 Visoka karta
# 
# Za razlago se obrnite na
# [Wikipedijo](https://en.wikipedia.org/wiki/List_of_poker_hands).
# 
#     >>> vrednost([(10, 'križ'), (12, 'srce'), (12, 'pik'), (10, 'kara'), (12, 'križ')])
#     7
# =============================================================================

# =====================================================================@024236=
# 9. podnaloga
# Sestavite funkcijo `ovrednoti(karte)`, ki sprejme seznam kart (dolžine vsaj
# pet) in vrne vrednost najboljše peterke v seznamu.
# 
# Pomagate si lahko s funkcijo `combinations` iz modula `itertools`.
# =============================================================================

# =====================================================================@024237=
# 10. podnaloga
# Sestavite funkcijo `poker(imena)`, ki ustvari nov kup kart, jih premeša, razdeli
# $n$-tim igralcem in odpre še skupne karte. Funkcija naj izpiše skupne karte,
# hkrati pa za vsakega igralca še njegovo ime, število točk in njegovi karti.
# 
#     >>> poker(["Ana", "Bine", "Cene"])
#     [(4, 'kara'), (10, 'križ'), (12, 'srce'), (8, 'križ'), (12, 'pik')]
#     Ana 3 [(13, 'križ'), (8, 'kara')]
#     Bine 3 [(4, 'križ'), (2, 'srce')]
#     Cene 7 [(10, 'kara'), (12, 'križ')]
# =============================================================================






































































































# ============================================================================@
# fmt: off
"Če vam Python sporoča, da je v tej vrstici sintaktična napaka,"
"se napaka v resnici skriva v zadnjih vrsticah vaše kode."

"Kode od tu naprej NE SPREMINJAJTE!"

# isort: off
import json
import os
import re
import shutil
import sys
import traceback
import urllib.error
import urllib.request
import io
from contextlib import contextmanager


class VisibleStringIO(io.StringIO):
    def read(self, size=None):
        x = io.StringIO.read(self, size)
        print(x, end="")
        return x

    def readline(self, size=None):
        line = io.StringIO.readline(self, size)
        print(line, end="")
        return line


class TimeoutError(Exception):
    pass


class Check:
    parts = None
    current_part = None
    part_counter = None

    @staticmethod
    def has_solution(part):
        return part["solution"].strip() != ""

    @staticmethod
    def initialize(parts):
        Check.parts = parts
        for part in Check.parts:
            part["valid"] = True
            part["feedback"] = []
            part["secret"] = []

    @staticmethod
    def part():
        if Check.part_counter is None:
            Check.part_counter = 0
        else:
            Check.part_counter += 1
        Check.current_part = Check.parts[Check.part_counter]
        return Check.has_solution(Check.current_part)

    @staticmethod
    def feedback(message, *args, **kwargs):
        Check.current_part["feedback"].append(message.format(*args, **kwargs))

    @staticmethod
    def error(message, *args, **kwargs):
        Check.current_part["valid"] = False
        Check.feedback(message, *args, **kwargs)

    @staticmethod
    def clean(x, digits=6, typed=False):
        t = type(x)
        if t is float:
            x = round(x, digits)
            # Since -0.0 differs from 0.0 even after rounding,
            # we change it to 0.0 abusing the fact it behaves as False.
            v = x if x else 0.0
        elif t is complex:
            v = complex(
                Check.clean(x.real, digits, typed), Check.clean(x.imag, digits, typed)
            )
        elif t is list:
            v = list([Check.clean(y, digits, typed) for y in x])
        elif t is tuple:
            v = tuple([Check.clean(y, digits, typed) for y in x])
        elif t is dict:
            v = sorted(
                [
                    (Check.clean(k, digits, typed), Check.clean(v, digits, typed))
                    for (k, v) in x.items()
                ]
            )
        elif t is set:
            v = sorted([Check.clean(y, digits, typed) for y in x])
        else:
            v = x
        return (t, v) if typed else v

    @staticmethod
    def secret(x, hint=None, clean=None):
        clean = Check.get("clean", clean)
        Check.current_part["secret"].append((str(clean(x)), hint))

    @staticmethod
    def equal(expression, expected_result, clean=None, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        actual_result = eval(expression, global_env)
        if clean(actual_result) != clean(expected_result):
            Check.error(
                "Izraz {0} vrne {1!r} namesto {2!r}.",
                expression,
                actual_result,
                expected_result,
            )
            return False
        else:
            return True

    @staticmethod
    def approx(expression, expected_result, tol=1e-6, env=None, update_env=None):
        try:
            import numpy as np
        except ImportError:
            Check.error("Namestiti morate numpy.")
            return False
        if not isinstance(expected_result, np.ndarray):
            Check.error("Ta funkcija je namenjena testiranju za tip np.ndarray.")

        if env is None:
            env = dict()
        env.update({"np": np})
        global_env = Check.init_environment(env=env, update_env=update_env)
        actual_result = eval(expression, global_env)
        if type(actual_result) is not type(expected_result):
            Check.error(
                "Rezultat ima napačen tip. Pričakovan tip: {}, dobljen tip: {}.",
                type(expected_result).__name__,
                type(actual_result).__name__,
            )
            return False
        exp_shape = expected_result.shape
        act_shape = actual_result.shape
        if exp_shape != act_shape:
            Check.error(
                "Obliki se ne ujemata. Pričakovana oblika: {}, dobljena oblika: {}.",
                exp_shape,
                act_shape,
            )
            return False
        try:
            np.testing.assert_allclose(
                expected_result, actual_result, atol=tol, rtol=tol
            )
            return True
        except AssertionError as e:
            Check.error("Rezultat ni pravilen." + str(e))
            return False

    @staticmethod
    def run(statements, expected_state, clean=None, env=None, update_env=None):
        code = "\n".join(statements)
        statements = "  >>> " + "\n  >>> ".join(statements)
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        exec(code, global_env)
        errors = []
        for x, v in expected_state.items():
            if x not in global_env:
                errors.append(
                    "morajo nastaviti spremenljivko {0}, vendar je ne".format(x)
                )
            elif clean(global_env[x]) != clean(v):
                errors.append(
                    "nastavijo {0} na {1!r} namesto na {2!r}".format(
                        x, global_env[x], v
                    )
                )
        if errors:
            Check.error("Ukazi\n{0}\n{1}.", statements, ";\n".join(errors))
            return False
        else:
            return True

    @staticmethod
    @contextmanager
    def in_file(filename, content, encoding=None):
        encoding = Check.get("encoding", encoding)
        with open(filename, "w", encoding=encoding) as f:
            for line in content:
                print(line, file=f)
        old_feedback = Check.current_part["feedback"][:]
        yield
        new_feedback = Check.current_part["feedback"][len(old_feedback) :]
        Check.current_part["feedback"] = old_feedback
        if new_feedback:
            new_feedback = ["\n    ".join(error.split("\n")) for error in new_feedback]
            Check.error(
                "Pri vhodni datoteki {0} z vsebino\n  {1}\nso se pojavile naslednje napake:\n- {2}",
                filename,
                "\n  ".join(content),
                "\n- ".join(new_feedback),
            )

    @staticmethod
    @contextmanager
    def input(content, visible=None):
        old_stdin = sys.stdin
        old_feedback = Check.current_part["feedback"][:]
        try:
            with Check.set_stringio(visible):
                sys.stdin = Check.get("stringio")("\n".join(content) + "\n")
                yield
        finally:
            sys.stdin = old_stdin
        new_feedback = Check.current_part["feedback"][len(old_feedback) :]
        Check.current_part["feedback"] = old_feedback
        if new_feedback:
            new_feedback = ["\n  ".join(error.split("\n")) for error in new_feedback]
            Check.error(
                "Pri vhodu\n  {0}\nso se pojavile naslednje napake:\n- {1}",
                "\n  ".join(content),
                "\n- ".join(new_feedback),
            )

    @staticmethod
    def out_file(filename, content, encoding=None):
        encoding = Check.get("encoding", encoding)
        with open(filename, encoding=encoding) as f:
            out_lines = f.readlines()
        equal, diff, line_width = Check.difflines(out_lines, content)
        if equal:
            return True
        else:
            Check.error(
                "Izhodna datoteka {0}\n  je enaka{1}  namesto:\n  {2}",
                filename,
                (line_width - 7) * " ",
                "\n  ".join(diff),
            )
            return False

    @staticmethod
    def output(expression, content, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        too_many_read_requests = False
        try:
            exec(expression, global_env)
        except EOFError:
            too_many_read_requests = True
        finally:
            output = sys.stdout.getvalue().rstrip().splitlines()
            sys.stdout = old_stdout
        equal, diff, line_width = Check.difflines(output, content)
        if equal and not too_many_read_requests:
            return True
        else:
            if too_many_read_requests:
                Check.error("Program prevečkrat zahteva uporabnikov vnos.")
            if not equal:
                Check.error(
                    "Program izpiše{0}  namesto:\n  {1}",
                    (line_width - 13) * " ",
                    "\n  ".join(diff),
                )
            return False

    @staticmethod
    def difflines(actual_lines, expected_lines):
        actual_len, expected_len = len(actual_lines), len(expected_lines)
        if actual_len < expected_len:
            actual_lines += (expected_len - actual_len) * ["\n"]
        else:
            expected_lines += (actual_len - expected_len) * ["\n"]
        equal = True
        line_width = max(
            len(actual_line.rstrip())
            for actual_line in actual_lines + ["Program izpiše"]
        )
        diff = []
        for out, given in zip(actual_lines, expected_lines):
            out, given = out.rstrip(), given.rstrip()
            if out != given:
                equal = False
            diff.append(
                "{0} {1} {2}".format(
                    out.ljust(line_width), "|" if out == given else "*", given
                )
            )
        return equal, diff, line_width

    @staticmethod
    def init_environment(env=None, update_env=None):
        global_env = globals()
        if not Check.get("update_env", update_env):
            global_env = dict(global_env)
        global_env.update(Check.get("env", env))
        return global_env

    @staticmethod
    def generator(
        expression,
        expected_values,
        should_stop=None,
        further_iter=None,
        clean=None,
        env=None,
        update_env=None,
    ):
        from types import GeneratorType

        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        gen = eval(expression, global_env)
        if not isinstance(gen, GeneratorType):
            Check.error("Izraz {0} ni generator.", expression)
            return False

        try:
            for iteration, expected_value in enumerate(expected_values):
                actual_value = next(gen)
                if clean(actual_value) != clean(expected_value):
                    Check.error(
                        "Vrednost #{0}, ki jo vrne generator {1} je {2!r} namesto {3!r}.",
                        iteration,
                        expression,
                        actual_value,
                        expected_value,
                    )
                    return False
            for _ in range(Check.get("further_iter", further_iter)):
                next(gen)  # we will not validate it
        except StopIteration:
            Check.error("Generator {0} se prehitro izteče.", expression)
            return False

        if Check.get("should_stop", should_stop):
            try:
                next(gen)
                Check.error("Generator {0} se ne izteče (dovolj zgodaj).", expression)
            except StopIteration:
                pass  # this is fine
        return True

    @staticmethod
    def summarize():
        for i, part in enumerate(Check.parts):
            if not Check.has_solution(part):
                print("{0}. podnaloga je brez rešitve.".format(i + 1))
            elif not part["valid"]:
                print("{0}. podnaloga nima veljavne rešitve.".format(i + 1))
            else:
                print("{0}. podnaloga ima veljavno rešitev.".format(i + 1))
            for message in part["feedback"]:
                print("  - {0}".format("\n    ".join(message.splitlines())))

    settings_stack = [
        {
            "clean": clean.__func__,
            "encoding": None,
            "env": {},
            "further_iter": 0,
            "should_stop": False,
            "stringio": VisibleStringIO,
            "update_env": False,
        }
    ]

    @staticmethod
    def get(key, value=None):
        if value is None:
            return Check.settings_stack[-1][key]
        return value

    @staticmethod
    @contextmanager
    def set(**kwargs):
        settings = dict(Check.settings_stack[-1])
        settings.update(kwargs)
        Check.settings_stack.append(settings)
        try:
            yield
        finally:
            Check.settings_stack.pop()

    @staticmethod
    @contextmanager
    def set_clean(clean=None, **kwargs):
        clean = clean or Check.clean
        with Check.set(clean=(lambda x: clean(x, **kwargs)) if kwargs else clean):
            yield

    @staticmethod
    @contextmanager
    def set_environment(**kwargs):
        env = dict(Check.get("env"))
        env.update(kwargs)
        with Check.set(env=env):
            yield

    @staticmethod
    @contextmanager
    def set_stringio(stringio):
        if stringio is True:
            stringio = VisibleStringIO
        elif stringio is False:
            stringio = io.StringIO
        if stringio is None or stringio is Check.get("stringio"):
            yield
        else:
            with Check.set(stringio=stringio):
                yield

    @staticmethod
    @contextmanager
    def time_limit(timeout_seconds=1):
        from signal import SIGINT, raise_signal
        from threading import Timer

        def interrupt_main():
            raise_signal(SIGINT)

        timer = Timer(timeout_seconds, interrupt_main)
        timer.start()
        try:
            yield
        except KeyboardInterrupt:
            raise TimeoutError
        finally:
            timer.cancel()


def _validate_current_file():
    def extract_parts(filename):
        with open(filename, encoding="utf-8") as f:
            source = f.read()
        part_regex = re.compile(
            r"# =+@(?P<part>\d+)=\s*\n"  # beginning of header
            r"(\s*#( [^\n]*)?\n)+?"  # description
            r"\s*# =+\s*?\n"  # end of header
            r"(?P<solution>.*?)"  # solution
            r"(?=\n\s*# =+@)",  # beginning of next part
            flags=re.DOTALL | re.MULTILINE,
        )
        parts = [
            {"part": int(match.group("part")), "solution": match.group("solution")}
            for match in part_regex.finditer(source)
        ]
        # The last solution extends all the way to the validation code,
        # so we strip any trailing whitespace from it.
        parts[-1]["solution"] = parts[-1]["solution"].rstrip()
        return parts

    def backup(filename):
        backup_filename = None
        suffix = 1
        while not backup_filename or os.path.exists(backup_filename):
            backup_filename = "{0}.{1}".format(filename, suffix)
            suffix += 1
        shutil.copy(filename, backup_filename)
        return backup_filename

    def submit_parts(parts, url, token):
        submitted_parts = []
        for part in parts:
            if Check.has_solution(part):
                submitted_part = {
                    "part": part["part"],
                    "solution": part["solution"],
                    "valid": part["valid"],
                    "secret": [x for (x, _) in part["secret"]],
                    "feedback": json.dumps(part["feedback"]),
                }
                if "token" in part:
                    submitted_part["token"] = part["token"]
                submitted_parts.append(submitted_part)
        data = json.dumps(submitted_parts).encode("utf-8")
        headers = {"Authorization": token, "content-type": "application/json"}
        request = urllib.request.Request(url, data=data, headers=headers)
        # This is a workaround because some clients (and not macOS ones!) report
        # <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1129)>
        import ssl

        context = ssl._create_unverified_context()
        response = urllib.request.urlopen(request, context=context)
        # When the issue is resolved, the following should be used
        # response = urllib.request.urlopen(request)
        return json.loads(response.read().decode("utf-8"))

    def update_attempts(old_parts, response):
        updates = {}
        for part in response["attempts"]:
            part["feedback"] = json.loads(part["feedback"])
            updates[part["part"]] = part
        for part in old_parts:
            valid_before = part["valid"]
            part.update(updates.get(part["part"], {}))
            valid_after = part["valid"]
            if valid_before and not valid_after:
                wrong_index = response["wrong_indices"].get(str(part["part"]))
                if wrong_index is not None:
                    hint = part["secret"][wrong_index][1]
                    if hint:
                        part["feedback"].append("Namig: {}".format(hint))

    filename = os.path.abspath(sys.argv[0])
    file_parts = extract_parts(filename)
    Check.initialize(file_parts)

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0IjoyNDIyOCwidXNlciI6MTA2NzZ9:1u1SqN:nnNkLAOWt1zJ1kLymxMN6ZJLUGyzJ5dRmC5h-qPNCu8"
        try:
            for stevilka in range(2, 15):
                for barva in ["srce", "kara", "pik", "križ"]:
                    Check.equal("({}, '{}') in nov_kup()".format(stevilka, barva), True)
            
            Check.equal("len(nov_kup())", 52)
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0IjoyNDIyOSwidXNlciI6MTA2NzZ9:1u1SqN:XGEqKqVvADALYXNetOJLHR051XVQ1ERI9UI-YV_ZWlI"
        try:
            def check_premesaj(karte):
                karte_input = karte[:]
                found_permutation = False
                for _ in range(50):
                    karte_trenutno = karte[:]
                    premesaj(karte)
                    if not isinstance(karte, list) or len(karte_trenutno) != len(karte):
                        Check.error(
                            f"Klic premesaj({karte_trenutno})\nni zgolj premešal seznama. "
                            f"Trenutno stanje kart: {karte}\n"
                        )
                        return False
                    if karte_input != karte:
                        found_permutation = True
                if not found_permutation:
                    Check.error(
                        f"50 zaporednih klicev premesaj({karte_input})\nni premešalo kart. "
                        f"Z veliko verjetnostjo trdimo, da je s funkcijo premesaj nekaj narobe."
                    )
            
            
            
            cases = [
                [(2, "srce"), (3, "srce"), (4, "srce"), (5, "srce")],
                [(2, "kara"), (3, "pik"), (4, "srce"), (5, "srce")]
            ]
            for case in cases:
                check_premesaj(case)
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0IjoyNDIzMCwidXNlciI6MTA2NzZ9:1u1SqN:Pvnw4--tjoqyRPx4x3LvdwzVHyH-yoOBBoDdBdYYikE"
        try:
            def check_razdeli_karte(igralci, karte):
                odgovor = razdeli_karte(igralci, karte)
                if type(odgovor) != dict:
                    Check.error(f"razdeli_karte({igralci}, {karte})\nne vrne slovarja.")
                    return False
                if len(odgovor) != len(igralci):
                    Check.error(
                        f"Po klicu razdeli_karte({igralci}, {karte})\nje število igralcev s kartami {len(odgovor)}, "
                        f"število igralcev je pa {len(igralci)}."
                    )
                    return False
                if not all(len(karti) == 2 for karti in razdeli_karte(["Ana", "Bine", "Cene"], nov_kup()).values()):
                    Check.error(f"Po klicu razdeli_karte({igralci}, {karte})\nnimajo vsi igralci dveh kart")
                    return False
                different_cards = len(
                    set(karta for karti in razdeli_karte(["Ana", "Bine", "Cene"], nov_kup()).values() for karta in karti)
                )
                expected_different_cards = 2 * len(igralci)
                if different_cards != 2 * len(igralci):
                    Check.error(
                        f"Po klicu razdeli_karte({igralci}, {karte})\n"
                        f"se skupno število različnih kart ({different_cards}) "
                        f"ne ujema s pričakovanim ({expected_different_cards})."
                    )
                return True
            
            
            check_razdeli_karte(["Ana", "Bine", "Cene"], nov_kup())
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0IjoyNDIzMSwidXNlciI6MTA2NzZ9:1u1SqN:LQWRM4xJ2MYFd4xRaQkPweWVUF5s7QoE5X-4kwOht5E"
        try:
            cases = [
                (
                    [(2, "kara"), (3, "kara"), (4, "srce"), (6, "križ"), (8, "pik"), (10, "kara")],
                    [(3, "kara"), (4, "srce"), (6, "križ"), (8, "pik"), (10, "kara")]
                ),
                (
                    [(2, "kara"), (3, "kara"), (4, "srce"), (6, "križ"), (8, "pik")],
                    [(2, "kara"), (3, "kara"), (4, "srce"), (6, "križ"), (8, "pik")],
                )
            ]
            
            def check_odpri_skupne_karte(karte, expected_output):
                karte_input = karte[:]
                answer = odpri_skupne_karte(karte)
                if not isinstance(answer, list):
                    Check.error(f"Klic odpri_skupne_karte({karte_input})\nmora vrniti seznam.")
                    return False
                answer = sorted(answer)
                if answer != expected_output:
                    Check.error(f"Klic odpri_skupne_karte({karte_input})\nmora vrniti (permutacijo) {expected_output}, "
                                f"a vrne {answer}")
                    return False
                if len(karte_input) != len(answer) + len(karte):
                    Check.error(f"Klic odpri_skupne_karte({karte_input})\nni odstranil kart z začetnega kupa.")
                    return False
                return True
            
            
            for karte, cards_out in cases:
                check_odpri_skupne_karte(karte, cards_out)
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0IjoyNDIzMiwidXNlciI6MTA2NzZ9:1u1SqN:zyXANLWI1AVPb2K6aZLKs603zBSR9QW-kBDjqQFH4AA"
        try:
            Check.equal("na_dva_dela([(10, 'križ'), (12, 'srce'), (12, 'pik'), (10, 'kara'), (12, 'križ')])",
            ([10, 12, 12, 10, 12], ['križ', 'srce', 'pik', 'kara', 'križ']))
            
            Check.equal("na_dva_dela([(2, 'kara'), (4, 'kara'), (10, 'kara'), (3, 'pik'), (9, 'križ')])",
            ([2, 4, 10, 3, 9], ['kara', 'kara', 'kara', 'pik', 'križ']))
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0IjoyNDIzMywidXNlciI6MTA2NzZ9:1u1SqN:I9nSFsVLGPbAJnHTYxyPdixVzZhHBT677hG44ytLcSk"
        try:
            Check.equal("tvorijo_lestvico([(10, 'križ'), (12, 'srce')])", False)
            Check.equal("tvorijo_lestvico([(10, 'križ'), (12, 'srce'), (11, 'križ')])", True)
            Check.equal("tvorijo_lestvico([(10, 'križ'), (10, 'srce')])", False)
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0IjoyNDIzNCwidXNlciI6MTA2NzZ9:1u1SqN:6UA5YlDMAd7RZHcSgFFR7MCKR_tOxiubbwhwrbFwc40"
        try:
            Check.equal("kolikokrat_se_pojavi_katera_stevilka([(10, 'križ'), (12, 'srce'), (12, 'pik'), (10, 'kara'), (12, 'križ')])",
            {10: 2, 12: 3})
            
            Check.equal("kolikokrat_se_pojavi_katera_stevilka([(2, 'kara'), (4, 'kara'), (10, 'kara'), (3, 'pik'), (9, 'križ')])",
            {9: 1, 2: 1, 3: 1, 4: 1, 10: 1})
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0IjoyNDIzNSwidXNlciI6MTA2NzZ9:1u1SqN:9kNC-bskvTi04X3zTmWY1eX_WqT4SVR5MilwOyEhBO8"
        try:
            Check.equal("vrednost([(10, 'križ'), (5, 'kara'), (4, 'križ'), (7, 'pik'), (11, 'pik')])", 1)
            Check.equal("vrednost([(4, 'kara'), (2, 'križ'), (7, 'križ'), (7, 'kara'), (13, 'križ')])", 2)
            Check.equal("vrednost([(5, 'pik'), (12, 'srce'), (13, 'kara'), (14, 'srce'), (5, 'srce')])", 2)
            Check.equal("vrednost([(8, 'križ'), (9, 'kara'), (5, 'križ'), (8, 'pik'), (3, 'srce')])", 2)
            Check.equal("vrednost([(11, 'kara'), (2, 'srce'), (2, 'pik'), (3, 'križ'), (4, 'pik')])", 2)
            Check.equal("vrednost([(6, 'kara'), (10, 'pik'), (13, 'pik'), (12, 'pik'), (7, 'srce')])", 1)
            Check.equal("vrednost([(9, 'pik'), (3, 'pik'), (10, 'srce'), (11, 'srce'), (2, 'kara')])", 1)
            Check.equal("vrednost([(11, 'križ'), (14, 'kara'), (12, 'kara'), (14, 'pik'), (6, 'srce')])", 2)
            Check.equal("vrednost([(9, 'srce'), (6, 'križ'), (6, 'pik'), (4, 'srce'), (13, 'srce')])", 2)
            Check.equal("vrednost([(3, 'kara'), (8, 'srce'), (10, 'kara'), (8, 'kara'), (12, 'križ')])", 2)
            Check.equal("vrednost([(14, 'križ'), (14, 'pik'), (14, 'kara'), (14, 'srce'), (13, 'križ')])", 8)
            Check.equal("vrednost([(13, 'pik'), (13, 'kara'), (13, 'srce'), (12, 'križ'), (12, 'pik')])", 7)
            Check.equal("vrednost([(12, 'kara'), (12, 'srce'), (11, 'križ'), (11, 'pik'), (11, 'kara')])", 7)
            Check.equal("vrednost([(11, 'srce'), (10, 'križ'), (10, 'pik'), (10, 'kara'), (10, 'srce')])", 8)
            Check.equal("vrednost([(9, 'križ'), (9, 'pik'), (9, 'kara'), (9, 'srce'), (8, 'križ')])", 8)
            Check.equal("vrednost([(8, 'pik'), (8, 'kara'), (8, 'srce'), (7, 'križ'), (7, 'pik')])", 7)
            Check.equal("vrednost([(7, 'kara'), (7, 'srce'), (6, 'križ'), (6, 'pik'), (6, 'kara')])", 7)
            Check.equal("vrednost([(6, 'srce'), (5, 'križ'), (5, 'pik'), (5, 'kara'), (5, 'srce')])", 8)
            Check.equal("vrednost([(4, 'križ'), (4, 'pik'), (4, 'kara'), (4, 'srce'), (3, 'križ')])", 8)
            Check.equal("vrednost([(3, 'pik'), (3, 'kara'), (3, 'srce'), (2, 'križ'), (2, 'pik')])", 7)
            Check.equal("vrednost([(4, 'pik'), (3, 'kara'), (2, 'srce'), (6, 'križ'), (5, 'pik')])", 5)
            Check.equal("vrednost([(4, 'pik'), (3, 'pik'), (7, 'pik'), (6, 'pik'), (5, 'pik')])", 9)
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0IjoyNDIzNiwidXNlciI6MTA2NzZ9:1u1SqN:A-hbglB4lGZXg1Tcw8ExztAmcZYbWDosHR1ZcalQ9E4"
        try:
            Check.equal("ovrednoti([(8, 'križ'), (10, 'srce'), (13, 'križ'), (7, 'pik'), (3, 'kara')])", 1)
            Check.equal("ovrednoti([(6, 'pik'), (10, 'pik'), (8, 'križ'), (5, 'kara'), (14, 'pik'), (10, 'kara')])", 2)
            Check.equal("ovrednoti([(14, 'kara'), (12, 'križ'), (10, 'kara'), (14, 'srce'), (5, 'kara'), (13, 'križ'), (8, 'srce')])", 2)
            Check.equal("ovrednoti([(5, 'srce'), (5, 'kara'), (12, 'križ'), (5, 'pik'), (14, 'pik'), (4, 'pik'), (11, 'kara'), (14, 'srce')])", 7)
            Check.equal("ovrednoti([(13, 'pik'), (5, 'kara'), (13, 'križ'), (3, 'križ'), (10, 'križ'), (14, 'pik'), (7, 'srce'), (5, 'srce'), (8, 'kara')])", 3)
            Check.equal("ovrednoti([(3, 'križ'), (7, 'pik'), (2, 'pik'), (3, 'srce'), (5, 'kara'), (4, 'križ'), (4, 'pik'), (6, 'kara'), (12, 'pik'), (14, 'pik')])", 6)
            Check.equal("ovrednoti([(2, 'križ'), (11, 'kara'), (14, 'srce'), (3, 'križ'), (6, 'križ'), (6, 'kara'), (5, 'križ'), (13, 'križ'), (4, 'križ'), (2, 'kara'), (12, 'križ')])", 9)
            Check.equal("ovrednoti([(9, 'kara'), (2, 'pik'), (4, 'pik'), (8, 'kara'), (13, 'pik'), (9, 'križ'), (11, 'kara'), (13, 'srce'), (4, 'križ'), (9, 'srce'), (5, 'križ'), (10, 'križ')])", 7)
            Check.equal("ovrednoti([(3, 'srce'), (11, 'kara'), (6, 'kara'), (6, 'križ'), (10, 'srce'), (5, 'križ'), (12, 'pik'), (7, 'kara'), (5, 'kara'), (3, 'kara'), (3, 'križ'), (6, 'pik'), (13, 'srce')])", 7)
            Check.equal("ovrednoti([(13, 'srce'), (2, 'kara'), (10, 'pik'), (9, 'srce'), (14, 'srce'), (14, 'kara'), (6, 'kara'), (9, 'križ'), (14, 'križ'), (12, 'srce'), (11, 'srce'), (3, 'srce'), (12, 'križ'), (9, 'kara')])", 7)
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0IjoyNDIzNywidXNlciI6MTA2NzZ9:1u1SqN:jgy_7shSgcu8bhDVatSALf5PtkJxjHFMYpgCxOk8xxs"
        try:
            pass
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    print("Shranjujem rešitve na strežnik... ", end="")
    try:
        url = "https://www.projekt-tomo.si/api/attempts/submit/"
        token = "Token 6c1427226c4491e4c9508056163a2002efbcc7bf"
        response = submit_parts(Check.parts, url, token)
    except urllib.error.URLError:
        message = (
            "\n"
            "-------------------------------------------------------------------\n"
            "PRI SHRANJEVANJU JE PRIŠLO DO NAPAKE!\n"
            "Preberite napako in poskusite znova ali se posvetujte z asistentom.\n"
            "-------------------------------------------------------------------\n"
        )
        print(message)
        traceback.print_exc()
        print(message)
        sys.exit(1)
    else:
        print("Rešitve so shranjene.")
        update_attempts(Check.parts, response)
        if "update" in response:
            print("Updating file... ", end="")
            backup_filename = backup(filename)
            with open(__file__, "w", encoding="utf-8") as f:
                f.write(response["update"])
            print("Previous file has been renamed to {0}.".format(backup_filename))
            print("If the file did not refresh in your editor, close and reopen it.")
    Check.summarize()


if __name__ == "__main__":
    _validate_current_file()
