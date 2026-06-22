# -*- coding: utf-8 -*-
# =============================================================================
# [単語マスク対応表] テキスト置換でオリジナルに戻せます
#
#   公開版 (本ファイル)  →  オリジナル
#   ----------------------------------------
#   UNIT                 →  SSD
#   PART                 →  CHIP
#   batch                →  lot
#   anomaly              →  defect
#   filtered / filtering →  screened / screening
#   release              →  retail
# =============================================================================
"""
商品フィルタリング分析 - 期間別統計 (拡張版評価指標付)

このツールはUNIT製品のフィルタリング分析を実行します。
複数期間スライディングウィンドウ分析、期間別スライディングウィンドウ分析
SMOTE (Synthetic Minority Over-sampling Technique) による不均衡データ対応

別の機能:
- PARTレベルの分析予測複数予測
- 期間別スライディングウィンドウ分析
- SMOTE (Synthetic Minority Over-sampling Technique) による不均衡データ対応

作成者: masatttt
作成日: 2026-06-17
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
import os
import sys
from datetime import datetime

# 利用可能なモデル定義
MODEL_REGISTRY = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=1, max_iter=1000),
    'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=1, n_estimators=100),
    'ExtraTrees': ExtraTreesClassifier(class_weight='balanced', random_state=1, n_estimators=100),
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=1, max_iter=100),
}

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings('ignore')


class EnhancedPeriodScreeningAnalysis:
    """
    商品フィルタリングの分析クラス

    このクラスは以下の分析機能を提供します:
    1. PARTレベルの分析予測 (複数期間に渡る)
    2. UNIT規格での予測 (月次・四半期)
    3. ロジスティック回帰モデルの訓練
    4. 期間別スライディングウィンドウ分析
    5. 期間別統計分析
    """

    def __init__(self, threshold_values=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5),
                 use_smote=True, use_smote_neighbors=5, slide_out_mapping=None,
                 models=None):
        """
        Parameters:
            threshold_values (list): 評価に使用する固定閾値のリスト
            use_smote (bool): SMOTE (不均衡データ処理) の使用 (デフォルト True)
            models (list or None): 使用するモデル名リスト。Noneなら全モデル。
                選択肢: 'LogisticRegression', 'RandomForest', 'ExtraTrees', 'HistGradientBoosting'
        """
        self.threshold_values = list(threshold_values)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.use_smote = use_smote
        self.use_smote_neighbors = use_smote_neighbors
        self.part_unit_mapping = None  # PARTからUNITへのマッピング辞書

        # 使用するモデルを選択
        import copy
        if models is None:
            self.models = {name: copy.deepcopy(m) for name, m in MODEL_REGISTRY.items()}
        else:
            unknown = [m for m in models if m not in MODEL_REGISTRY]
            if unknown:
                raise ValueError(f"未知のモデル名: {unknown}。選択肢: {list(MODEL_REGISTRY.keys())}")
            self.models = {name: copy.deepcopy(MODEL_REGISTRY[name]) for name in models}
        print(f"使用モデル: {list(self.models.keys())}")

    def load_part_unit_mapping(self, mapping_path):
        """
        PARTレベルのマッピングファイルを読み込む

        Parameters:
            mapping_path (str): マッピングファイルパス

        Returns:
            DataFrame or None: 読み込んだデータ、失敗時はNone
        """
        try:
            mapping_df = pd.read_csv(mapping_path)
            required_cols = ['PART_ID', 'batch_number']
            if 'PART_ID' not in mapping_df.columns or 'batch_number' not in mapping_df.columns:
                missing_cols = [col for col in required_cols if col not in mapping_df.columns]
                print(f"警告: マッピングファイルに必須列がありません: {missing_cols}")
                return None

            if 'UNIT_ID' not in mapping_df.columns:
                print("警告: UNIT_ID列がマッピングファイルにありません")
                return mapping_df

            self.part_unit_mapping = mapping_df
            print(f"マッピングファイル読み込み完了: {len(mapping_df)} エントリ")
            return mapping_df

        except Exception as e:
            print(f"エラー: マッピングファイルの読み込みに失敗しました: {e}")
            return None

    def load_and_preprocess_data(self, data_path):
        """
        メインデータの前処理

        Parameters:
            data_path (str): メインデータファイルのパス

        Returns:
            DataFrame or None: 前処理済みデータ、エラー一時はNone
        """
        try:
            if not os.path.exists(data_path):
                print(f"警告: データファイルが存在しません: {data_path}")
                return None

            df = pd.read_csv(data_path)
            print(f"データ読み込み完了: {len(df)} サンプル")

        except FileNotFoundError:
            print(f"エラー: ファイルが見つかりません: {data_path}")
            return None
        except Exception as e:
            print(f"エラー: データ読み込み中に予期しないエラーが発生しました: {e}")
            return None

        required_cols = ['PART_ID', 'batch_number', 'y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"エラー: 必須列がありません: {missing_cols}")
            return None

        # 日付列を数値に変換
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'DATE' in col]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass

        # SRS_month / SRS_quarter 生成
        date_col = None
        for candidate in ['date', 'DATE', 'test_date', 'TEST_DATE']:
            if candidate in df.columns:
                date_col = candidate
                break

        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df['SRS_month'] = df[date_col].dt.to_period('M').astype(str)
            df['SRS_quarter'] = df[date_col].dt.to_period('Q').astype(str)
        else:
            # 既存の期間列を確認
            if 'SRS_month' not in df.columns:
                print("警告: 期間列 (SRS_month) が見つかりません")

        # ターゲット変数の確認
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        invalid_y = df['y'].isna() | ~df['y'].isin([0, 1])
        if invalid_y.sum() > 0:
            print(f"警告: ターゲット変数に無効な値が {invalid_y.sum()} 件あります。除外します。")
            df = df[~invalid_y].copy()

        if 'OOP_ID' not in df.columns:
            df['OOP_ID'] = np.nan

        print(f"前処理完了: {len(df)} サンプル, anomaly率: {df['y'].mean():.4f}")
        return df

    def create_sliding_windows(self, df, analysis_type='monthly', max_train_periods=4, feature_names=None):
        """
        複数期間スライディングウィンドウ分析

        機械学習の性能向上のために、追加訓練期間のデータを使って、
        分析精度を高める「スライディングウィンドウ分析」を実施します。

        Parameters:
            df (DataFrame): 入力データ
            analysis_type (str): 'monthly' (月次) or 'quarterly' (四半期)
            max_train_periods (int): 学習に使用する最大期間数 (デフォルト4)

        Returns:
            list: 各ウィンドウの評価結果を含む辞書リスト
        """
        if analysis_type == 'monthly':
            period_column = 'SRS_month'
            test_window = 4  # 4カ月先をテスト予測
        else:  # quarterly
            period_column = 'SRS_quarter'
            test_window = 1  # 1四半期先をテスト予測

        if period_column not in df.columns:
            print(f"エラー: 期間列 '{period_column}' が見つかりません")
            return []

        unique_periods = sorted(df[period_column].dropna().unique())
        print(f"利用可能な期間数: {len(unique_periods)} [{unique_periods}]")

        windows = []
        start_test_idx = 1

        for test_idx in range(start_test_idx, len(unique_periods)):
            test_period = unique_periods[test_idx]
            available_train_periods = unique_periods[:test_idx]
            num_train_periods = min(len(available_train_periods), max_train_periods)

            if num_train_periods <= 0:
                continue

            # 実際の訓練期間を選択
            train_periods = available_train_periods[-num_train_periods:]
            train_data = df[df[period_column].isin(train_periods)].copy()
            test_data = df[df[period_column] == test_period].copy()

            print(f"\n[ウィンドウ {test_idx}] テスト期間: {test_period}")
            print(f"  訓練期間: {list(train_periods)} ({len(train_data)} サンプル)")
            print(f"  テストデータ: {len(test_data)} サンプル")

            if len(train_data) == 0 or len(test_data) == 0:
                print("  スキップ: データが不足しています")
                continue

            # 前処理 (全モデル共通)
            prep = self._prepare_features(train_data, feature_names)
            if prep is None:
                print("  スキップ: 前処理に失敗しました")
                continue
            X_balanced, y_balanced, scaler, imputer, smote_info, feature_cols, X_clean, y_clean = prep

            # 各モデルで訓練・評価
            for model_name, model_proto in self.models.items():
                import copy
                model = copy.deepcopy(model_proto)
                try:
                    model.fit(X_balanced, y_balanced)
                except Exception as e:
                    print(f"  [{model_name}] 訓練失敗: {e}")
                    continue

                model_info = self._build_model_info(model, model_name, feature_cols,
                                                    smote_info, X_clean, y_clean)

                window_result = self.calculate_period_level_statistics(
                    model, test_data, scaler, imputer, model_info
                )

                if window_result:
                    window_result['model_name'] = model_name
                    window_result['test_period'] = test_period
                    window_result['train_periods'] = list(train_periods)
                    window_result['analysis_type'] = analysis_type
                    window_result['train_samples'] = len(train_data)
                    window_result['test_samples'] = len(test_data)
                    window_result['train_anomaly_rate'] = float(train_data['y'].mean())
                    window_result['test_anomaly_rate'] = float(test_data['y'].mean())
                    windows.append(window_result)

        return windows

    def _prepare_features(self, train_data, feature_names=None):
        """
        全モデル共通の前処理: 特徴量選択・欠損値補完・スケーリング・SMOTE

        Returns:
            tuple: (X_balanced, y_balanced, scaler, imputer, smote_info, feature_cols, X_clean, y_clean)
                   or None on failure
        """
        exclude_cols = ['y', 'PART_ID', 'batch_number', 'UNIT_ID', 'OOP_ID',
                        'SRS_month', 'SRS_quarter', 'date', 'DATE']
        if feature_names is not None:
            missing = [f for f in feature_names if f not in train_data.columns]
            if missing:
                print(f"警告: 指定した特徴量列がデータにありません: {missing}")
            feature_cols = [f for f in feature_names if f in train_data.columns]
        else:
            feature_cols = [c for c in train_data.columns if c not in exclude_cols]

        if not feature_cols:
            print("エラー: 特徴量列がありません")
            return None

        X = train_data[feature_cols].copy()
        y = train_data['y'].copy()

        valid_mask = y.notna() & y.isin([0, 1])
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()

        if len(y_clean) == 0 or y_clean.nunique() < 2:
            print("エラー: 有効なクラスが不足しています")
            return None

        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_clean)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        smote_applied = False
        smote_samples_before = len(X_scaled)
        class_anomalies_before = int(y_clean.sum())
        class_anomalies_after = int(y_clean.sum())

        if self.use_smote:
            minority_count = int(y_clean.sum())
            if minority_count >= 2:
                k_neighbors = min(self.use_smote_neighbors, minority_count - 1)
                try:
                    smote_obj = SMOTE(random_state=1, k_neighbors=k_neighbors)
                    X_balanced, y_balanced = smote_obj.fit_resample(X_scaled, y_clean)
                    smote_applied = True
                    class_anomalies_after = int(y_balanced.sum())
                    print(f"  SMOTE適用: {smote_samples_before} → {len(X_balanced)} サンプル")
                except Exception as e:
                    print(f"  SMOTE失敗 (スキップ): {e}")
                    X_balanced, y_balanced = X_scaled, y_clean
            else:
                print("  SMOTEサンプリング後: 訓練データをそのまま使用します")
                X_balanced, y_balanced = X_scaled, y_clean
        else:
            X_balanced, y_balanced = X_scaled, y_clean

        smote_info = {
            'smote_applied': smote_applied,
            'smote_samples_before': smote_samples_before,
            'smote_samples_after': len(X_balanced),
            'class_anomalies_before': class_anomalies_before,
            'class_anomalies_after': class_anomalies_after,
        }

        return X_balanced, y_balanced, scaler, imputer, smote_info, feature_cols, X_clean, y_clean

    def _build_model_info(self, model, model_name, feature_cols, smote_info, X_clean, y_clean):
        """訓練済みモデルから model_info 辞書を構築する"""
        if hasattr(model, 'coef_'):
            feature_importance = abs(model.coef_[0]).tolist()
        elif hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_.tolist()
        else:
            feature_importance = []

        model_info = {
            'model_name': model_name,
            'feature_names': feature_cols,
            'feature_importance': feature_importance,
            'X_train': X_clean,
            'y_train': y_clean,
            **smote_info,
        }

        if model_name == 'LogisticRegression' and STATSMODELS_AVAILABLE:
            logit_info = self.fit_statsmodel_logit(X_clean, y_clean, feature_cols)
            if logit_info:
                model_info.update(logit_info)

        return model_info

    def train_model_with_thresholds(self, train_data, feature_names=None):
        """後方互換: LogisticRegression 単体で訓練する (旧インターフェース)"""
        import copy
        prep = self._prepare_features(train_data, feature_names)
        if prep is None:
            return None
        X_balanced, y_balanced, scaler, imputer, smote_info, feature_cols, X_clean, y_clean = prep

        model = copy.deepcopy(MODEL_REGISTRY['LogisticRegression'])
        model.fit(X_balanced, y_balanced)
        model_info = self._build_model_info(model, 'LogisticRegression', feature_cols,
                                            smote_info, X_clean, y_clean)
        return model, scaler, imputer, None, model_info

    def calculate_training_statistics(self, X_clean, y_clean, original_train_data):
        """
        訓練データの統計情報計算

        Parameters:
            X_clean (DataFrame): クリーニング済み訓練データ
            y_clean (Series): クリーニング済みターゲット変数
            original_train_data (DataFrame): 元の訓練データ

        Returns:
            dict: 統計情報辞書
        """
        total_samples = len(original_train_data)
        clean_samples = len(X_clean)
        anomaly_rate = float(y_clean.mean()) if len(y_clean) > 0 else 0.0

        # 特徴量統計
        feature_stats = {}
        for feature_name in X_clean.columns:
            feature_values = X_clean[feature_name].dropna()
            feature_stats[feature_name] = {
                'feature_values_mean': float(feature_values.mean()) if len(feature_values) > 0 else np.nan,
                'feature_values_std': float(feature_values.std()) if len(feature_values) > 0 else np.nan,
                'feature_values_min': float(feature_values.min()) if len(feature_values) > 0 else np.nan,
                'feature_values_median': float(feature_values.median()) if len(feature_values) > 0 else np.nan,
                'feature_values_max': float(feature_values.max()) if len(feature_values) > 0 else np.nan,
                'feature_values_null': int(X_clean[feature_name].isna().sum()),
            }

        # クラス分布
        class_distribution = {
            'class_0_count': int((y_clean == 0).sum()),
            'class_1_count': int((y_clean == 1).sum()),
            'class_0_ratio': float((y_clean == 0).mean()),
            'class_1_ratio': float((y_clean == 1).mean()),
        }

        return {
            'original_samples': total_samples,
            'clean_samples': clean_samples,
            'removed_samples': total_samples - clean_samples,
            'removal_rate': (total_samples - clean_samples) / total_samples if total_samples > 0 else 0,
            'feature_stats': feature_stats,
            'class_distribution': class_distribution,
            'anomaly_rate': anomaly_rate,
        }

    def fit_statsmodel_logit(self, X_clean, y_clean, feature_names):
        """
        statsmodels.Logitでデータを学習させる機能

        モデル係数やp値などの詳細な統計情報を得るために使用します。

        Returns:
            dict or None: 統計情報
        """
        if not STATSMODELS_AVAILABLE:
            return None

        try:
            imputer_tmp = SimpleImputer(strategy='median')
            X_imputed = imputer_tmp.fit_transform(X_clean)
            scaler_tmp = StandardScaler()
            X_scaled = scaler_tmp.fit_transform(X_imputed)
            X_sm = pd.DataFrame(X_scaled, columns=feature_names)
            X_sm = sm.add_constant(X_sm, has_constant='add')

            logit_model = sm.Logit(y_clean.astype(float).values, X_sm)
            logit_result = logit_model.fit(disp=False, maxiter=100)

            params = logit_result.params
            pvalues = logit_result.pvalues

            feature_coef = {}
            feature_pvalue = {}
            for fname in feature_names:
                if fname in params.index:
                    feature_coef[fname] = float(params[fname])
                    feature_pvalue[fname] = float(pvalues[fname])

            print("statsmodel logit結果: 係数・p値を取得しました")

            return {
                'intercept_coef': float(params.get('const', 0)),
                'intercept_pvalue': float(pvalues.get('const', 1)),
                'feature_coef': feature_coef,
                'feature_pvalue': feature_pvalue,
            }

        except Exception as e:
            print(f"statsmodel logit失敗: {e}")
            return None

    def calculate_percentile_threshold(self, model, training_info, scaler, imputer, percentile):
        """
        パーセンタイルから閾値を計算する

        Parameters:
            percentile: フィルタリング率 (0〜100)

        Returns:
            float: 閾値
        """
        X_train = training_info.get('X_train')
        if X_train is None or len(X_train) == 0:
            return 0.5

        feature_names = training_info.get('feature_names', [])
        X_feat = X_train[feature_names] if feature_names else X_train

        X_imputed = imputer.transform(X_feat)
        X_scaled = scaler.transform(X_imputed)
        probs = model.predict_proba(X_scaled)[:, 1]
        threshold = np.percentile(probs, 100 - percentile)
        return float(threshold)

    def calculate_abstract_threshold(self, model, training_info, scaler, imputer):
        """
        固定パーセンタイルでthresholdを計算する (固定値 9%)

        Returns:
            float: 閾値
        """
        return self.calculate_percentile_threshold(model, training_info, scaler, imputer, 9)

    def calculate_unit_level_statistics(self, test_data, y_pred, period_info):
        """
        UNITレベルの統計情報を計算する

        Parameters:
            test_data (DataFrame): 予測テストデータ
            y_pred (array): 予測結果
            period_info (dict): 期間情報

        Returns:
            dict: UNITレベル統計情報
        """
        if 'UNIT_ID' not in test_data.columns or test_data['UNIT_ID'].isna().all():
            return {
                'total_code': 0,
                'unit_with_parts': 0,
                'mapped_code': 0,
                'filtered_code': 0,
                'release_code': 0,
                'anomaly_code': 0,
                'filtered_code_rate': 0,
                'release_code_rate': 0,
                'unit_anomaly_rate': 0,
                'release_unit_anomaly_rate': 0,
            }

        unit_data = test_data.copy()
        unit_data['y_pred'] = y_pred
        unit_data_valid = unit_data[unit_data['UNIT_ID'].notna()].copy()

        if len(unit_data_valid) == 0:
            return {'total_code': 0, 'mapped_code': 0}

        # UNITレベルで集計
        unit_aggregated = unit_data_valid.groupby('UNIT_ID').agg(
            total_parts=('y', 'count'),
            anomaly_parts=('y', 'sum'),
            predicted_parts=('y_pred', 'sum'),
        ).reset_index()

        unit_aggregated['unit_filtered_flag'] = (unit_aggregated['predicted_parts'] > 0).astype(int)
        unit_aggregated['unit_has_anomaly'] = (unit_aggregated['anomaly_parts'] > 0).astype(int)

        # 統計計算
        total_code = len(unit_aggregated)
        mapped_code = total_code
        total_parts_in_code = int(unit_aggregated['total_parts'].sum())
        filtered_code = int(unit_aggregated['unit_filtered_flag'].sum())
        release_code = total_code - filtered_code
        anomaly_code = int(unit_aggregated['unit_has_anomaly'].sum())

        filtered_code_rate = filtered_code / total_code if total_code > 0 else 0
        release_code_rate = release_code / total_code if total_code > 0 else 0
        unit_anomaly_rate = anomaly_code / total_code if total_code > 0 else 0

        # リテール内のanomaly UNIT割合
        release_unit = unit_aggregated[unit_aggregated['unit_filtered_flag'] == 0]
        release_anomalyive = int((release_unit['unit_has_anomaly'] == 1).sum()) if len(release_unit) > 0 else 0
        release_unit_anomaly_rate = release_anomalyive / release_code if release_code > 0 else 0

        return {
            'total_code': total_code,
            'unit_with_parts': mapped_code,
            'mapped_code': mapped_code,
            'total_parts_in_code': total_parts_in_code,
            'filtered_code': filtered_code,
            'release_code': release_code,
            'anomaly_code': anomaly_code,
            'filtered_code_rate': filtered_code_rate,
            'release_code_rate': release_code_rate,
            'unit_anomaly_rate': unit_anomaly_rate,
            'release_unit_anomaly_rate': release_unit_anomaly_rate,
            'release_anomaly_code': release_anomalyive,
        }

    def calculate_period_level_statistics(self, model, test_data, scaler, imputer, training_info):
        """
        UNITレベルの評価統計を各閾値について計算する

        Parameters:
            model: 訓練済みモデル
            test_data (DataFrame): テストデータ
            scaler: スケーラー
            imputer: 欠損値補完器
            training_info (dict): 訓練情報

        Returns:
            dict: 評価結果
        """
        feature_names = training_info.get('feature_names', [])
        available_features = [f for f in feature_names if f in test_data.columns]

        if not available_features:
            print("エラー: テストデータに特徴量列がありません")
            return {}

        X_test = test_data[available_features].copy()
        y_true = test_data['y'].values

        X_test_imputed = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imputed)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        percentile_values = list(self.threshold_values)
        results_by_threshold = []

        for pct in percentile_values:
            threshold = self.calculate_percentile_threshold(model, training_info, scaler, imputer,
                                                            pct * 100 if pct <= 1 else pct)
            y_pred = (y_prob >= threshold).astype(int)

            total_samples = len(y_true)
            total_parts = total_samples
            total_anomalies = int(y_true.sum())

            if total_samples == 0:
                continue

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

            filtered_code = int(y_pred.sum())
            release_code = total_samples - filtered_code
            filtered_code_rate = filtered_code / total_samples if total_samples > 0 else 0

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            release_anomaly_rate = fn / release_code if release_code > 0 else 0
            release_anomaly_detection_rate = fn / total_anomalies if total_anomalies > 0 else 0

            try:
                auc = roc_auc_score(y_true, y_prob)
            except Exception:
                auc = np.nan

            # UNITレベル統計
            unit_stats = self.calculate_unit_level_statistics(test_data, y_pred, {})

            result = {
                'percentile_level': pct,
                'evaluation_level': 'top',
                'threshold': threshold,
                'total_samples': total_samples,
                'total_code': unit_stats.get('total_code', 0),
                'total_parts': total_parts,
                'total_anomalies': total_anomalies,
                'anomaly_parts': total_anomalies,
                'filtered_code': unit_stats.get('filtered_code', filtered_code),
                'filtered_code_rate': unit_stats.get('filtered_code_rate', filtered_code_rate),
                'release_code': unit_stats.get('release_code', release_code),
                'anomaly_code': unit_stats.get('anomaly_code', total_anomalies),
                'release_anomaly_rate': release_anomaly_rate,
                'release_anomaly_detection_rate': release_anomaly_detection_rate,
                'release_unit_anomaly_rate': unit_stats.get('release_unit_anomaly_rate', 0),
                'release_anomaly_code': unit_stats.get('release_anomaly_code', fn),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn),
                'test_sensitivity': sensitivity,
                'test_specificity': specificity,
                'test_precision': precision,
                'auc': auc,
                'unit_stats': unit_stats,
            }
            results_by_threshold.append(result)

        if not results_by_threshold:
            return {}

        return {
            'threshold_results': results_by_threshold,
            'y_prob': y_prob.tolist(),
            'auc': results_by_threshold[0].get('auc', np.nan),
        }

    def calculate_threshold(self, model, test_data, scaler, imputer, training_info):
        """
        固定閾値 (1%) のフィルタリング率で評価

        Returns:
            dict: 評価結果
        """
        return self.calculate_period_level_statistics(model, test_data, scaler, imputer, training_info)

    def run_analysis(self, data_path, mapping_path=None, output_dir=None,
                     analysis_type='monthly', max_train_periods=4, feature_names=None):
        """
        メイン分析を実行する

        Parameters:
            data_path (str): メインデータファイルパス
            mapping_path (str): マッピングファイルパス
            output_dir (str): 出力ディレクトリパス
            analysis_type (str): 'monthly' or 'quarterly'
            max_train_periods (int): 最大訓練期間数
        """
        print("=" * 60)
        print("UNIT商品フィルタリング分析 開始")
        print(f"分析タイプ: {analysis_type}, 最大訓練期間: {max_train_periods}")
        print("=" * 60)

        # データ読み込み
        df = self.load_and_preprocess_data(data_path)
        if df is None:
            print("エラー: データの読み込みに失敗しました。分析を終了します。")
            return None

        # マッピングファイルの読み込み
        if mapping_path:
            self.load_part_unit_mapping(mapping_path)

        # スライディングウィンドウ分析
        results = self.create_sliding_windows(df, analysis_type=analysis_type,
                                              max_train_periods=max_train_periods,
                                              feature_names=feature_names)

        if not results:
            print("警告: 分析結果がありません")
            return None

        # 結果の集計
        print(f"\n分析完了: {len(results)} ウィンドウ")

        # 出力ディレクトリの準備
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(data_path))

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 結果をDataFrameに変換して保存
        summary_rows = []
        for r in results:
            base = {k: v for k, v in r.items() if k not in ('threshold_results', 'y_prob', 'unit_stats')}
            if 'threshold_results' in r:
                for tr in r['threshold_results']:
                    row = {**base, **{k: v for k, v in tr.items() if k != 'unit_stats'}}
                    summary_rows.append(row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 全詳細結果
            summary_filename = os.path.join(output_dir,
                                            f"{analysis_type}_period_screening_{timestamp}.csv")
            summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
            print(f"ファイル保存: {summary_filename}")

            # モデル比較サマリー (モデル × 指標の平均)
            compare_cols = ['model_name', 'percentile_level', 'test_period', 'auc',
                            'test_sensitivity', 'test_specificity', 'test_precision',
                            'filtered_code_rate', 'release_anomaly_rate']
            available = [c for c in compare_cols if c in summary_df.columns]
            if 'model_name' in summary_df.columns and len(available) > 1:
                compare_df = (summary_df[available]
                              .groupby(['model_name', 'percentile_level', 'test_period'])
                              .mean()
                              .round(4)
                              .reset_index()
                              .sort_values(['model_name', 'test_period', 'percentile_level']))
                compare_filename = os.path.join(output_dir,
                                                f"{analysis_type}_model_comparison_{timestamp}.csv")
                compare_df.to_csv(compare_filename, index=False, encoding='utf-8-sig')
                print(f"モデル比較ファイル保存: {compare_filename}")

                # コンソールにサマリー表示 (AUC平均)
                if 'auc' in compare_df.columns:
                    auc_summary = (compare_df.groupby('model_name')['auc']
                                   .mean().sort_values(ascending=False))
                    print("\n=== モデル比較 (AUC平均) ===")
                    for name, auc in auc_summary.items():
                        print(f"  {name:<30} AUC={auc:.4f}")

        return results


def load_feature_file(feature_file):
    """
    特徴量ファイルを読み込む

    フォーマット: 1行1列名、# から始まる行はコメントとして無視

    Returns:
        list: 特徴量列名リスト
    """
    features = []
    with open(feature_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            features.append(line)
    if not features:
        raise ValueError(f"特徴量ファイルに有効な列名がありません: {feature_file}")
    print(f"特徴量ファイル読み込み完了: {len(features)} 列 → {features}")
    return features


def main():
    import argparse

    parser = argparse.ArgumentParser(description='UNIT商品フィルタリング分析')
    parser.add_argument('--data', required=True, help='データファイルパス (CSV)')
    parser.add_argument('--mapping', default=None, help='マッピングファイルパス (CSV)')
    parser.add_argument('--output', default=None, help='出力ディレクトリパス')
    parser.add_argument('--analysis', choices=['monthly', 'quarterly'], default='monthly',
                        help='分析タイプ (monthly or quarterly)')
    parser.add_argument('--max_periods', type=int, default=4, help='最大訓練期間数')
    parser.add_argument('--no_smote', action='store_true', help='SMOTEを使用しない')
    parser.add_argument('--feature_file', default=None,
                        help='使用する特徴量列名を記載したテキストファイル (1行1列名、#はコメント)')
    parser.add_argument('--models', nargs='+', default=None,
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f'使用するモデル名 (複数指定可)。省略時は全モデル。'
                             f'選択肢: {list(MODEL_REGISTRY.keys())}')
    args = parser.parse_args()

    # 特徴量リストの読み込み
    feature_names = None
    if args.feature_file:
        if not os.path.exists(args.feature_file):
            print(f"エラー: 特徴量ファイルが見つかりません: {args.feature_file}")
            sys.exit(1)
        feature_names = load_feature_file(args.feature_file)

    analyzer = EnhancedPeriodScreeningAnalysis(use_smote=not args.no_smote, models=args.models)
    results = analyzer.run_analysis(
        data_path=args.data,
        mapping_path=args.mapping,
        output_dir=args.output,
        analysis_type=args.analysis,
        max_train_periods=args.max_periods,
        feature_names=feature_names,
    )

    if results:
        print(f"\n分析完了: {len(results)} ウィンドウの結果を出力しました")
    else:
        print("分析に失敗しました")
        sys.exit(1)


if __name__ == '__main__':
    main()
