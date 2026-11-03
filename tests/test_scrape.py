import sqlite3
from main import update_horse_pages


class FakeSource:
    """成績ページの取得を模擬する。pages[horse_id] は「保存済みページ」と「最新ページ」に載っているレースIDの組"""
    def __init__(self, pages):
        self.pages = pages
        self.saved = {h: cached for h, (cached, _) in pages.items()}
        self.calls = []

    def get_horse_html(self, horse_id, force=False):
        self.calls.append((horse_id, force))
        if force or self.saved[horse_id] is None:
            self.saved[horse_id] = self.pages[horse_id][1]

    def get_ped_html(self, horse_id):
        pass


class FakeParser:
    def __init__(self, source):
        self.source = source
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE results (race_id TEXT, horse_id TEXT, rank TEXT)")
        self.conn.execute("CREATE TABLE horse_results (horse_id TEXT, race_id TEXT)")

    def parse_horse(self, horse_id):
        self.conn.execute("DELETE FROM horse_results WHERE horse_id = ?", (horse_id,))
        for race_id in self.source.saved[horse_id]:
            self.conn.execute("INSERT INTO horse_results VALUES (?, ?)", (horse_id, race_id))

    def parse_ped(self, horse_id):
        pass


def test_update_horse_pages_refetches_only_stale_pages(tmp_path):
    race = "202406050811"
    source = FakeSource({
        "fresh": ([race, "old"], [race, "old"]),   # 保存済みページに当該レースが載っている -> 再取得しない
        "stale": (["old"], [race, "old"]),          # レース前に保存された古いページ -> 再取得する
        "scratched": (["old"], ["old"]),            # 取消馬はページに載らない可能性があるので判定しない
        "new": (None, [race]),                      # 未保存 -> 通常の取得のみ
    })
    parser = FakeParser(source)
    parser.conn.executemany("INSERT INTO results VALUES (?, ?, ?)",
                            [(race, "fresh", "1"), (race, "stale", "2"), (race, "scratched", "取消"),
                             (race, "new", "3")])

    update_horse_pages(source, parser, race, str(tmp_path / "failures.tsv"))

    forced = [h for h, force in source.calls if force]
    assert forced == ["stale"]
    rows = set(parser.conn.execute("SELECT horse_id FROM horse_results WHERE race_id = ?", (race,)).fetchall())
    assert rows == {("fresh",), ("stale",), ("new",)}
