from abc import ABC, abstractmethod

class DataSource(ABC):
    """データ取得層の抽象インターフェース"""

    @abstractmethod
    def get_race_html(self, race_id: str) -> str:
        """指定されたrace_idのレース結果HTMLを取得する"""
        pass

    @abstractmethod
    def get_horse_html(self, horse_id: str) -> str:
        """指定されたhorse_idの馬成績HTMLを取得する"""
        pass

    @abstractmethod
    def get_ped_html(self, horse_id: str) -> str:
        """指定されたhorse_idの血統HTMLを取得する"""
        pass

    @abstractmethod
    def get_shutuba_html(self, race_id: str) -> str:
        """指定されたrace_idの出馬表HTMLを取得する"""
        pass
