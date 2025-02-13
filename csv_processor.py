from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import io

class CSVProcessor:
    def __init__(self, file_input):
        
        if isinstance(file_input, str):
            # ファイルパスを指定してデータフレームを作成
            self.df = pd.read_csv(file_input)
        
        else:
            byte_stream = file_input.read() # バイトストリームを取得
            decode_string = byte_stream.decode('utf-8') # バイトストリームをデコード
            string_io = io.StringIO(decode_string) # デコードした文字列をStringIOオブジェクトに変換
            self.df = pd.read_csv(string_io) # StringIOオブジェクトからデータフレームを作成
        
        self.encorders = {}
    
    def get_preview(self, rows=50):
        return self.df.head(rows).to_html(classes='table table-bordered table-striped', index=False)
    
    def get_basic_info(self):
        # データフレームの情報を取得
        info_buffer = io.StringIO() # StringIOオブジェクト作成
        self.df.info(buf=info_buffer) # データフレームの情報を取得
        return info_buffer.getvalue().replace("\n", "<br>") # 改行を<br>に変換
    
    def get_missing_info(self):
        # 欠損情報のデータフレーム作成
        total_rows = len(self.df) # データフレームの行数
        missing_count = self.df.isnull().sum() # 欠損数
        missing_percentage = (missing_count / total_rows * 100).round(2) # 欠損率

        missing_info_df = pd.DataFrame({
            '列名': missing_count.index,
            '欠損数': missing_count.values,
            '欠損率 (%)': missing_percentage.values
        })

        # 余計な空白を削除
        missing_info_df.columns = missing_info_df.columns.str.strip()
        return missing_info_df
    
    def highlight_missing_info(self, threshold=10):
        
        #欠損率に基づいて欠損値情報をハイライト表示
        missing_info_df = self.get_missing_info()

        def gradient_highlight(val):
            if isinstance(val, (int, float)):
                if val < threshold:
                    return ''  # ハイライトなし
                elif threshold <= val < 30:
                    return 'background-color: #ffe6e6;'  # 薄いピンク
                elif 30 <= val < 50:
                    return 'background-color: #ffcccc;'  # 薄い赤
                elif 50 <= val < 70:
                    return 'background-color: #ff9999;'  # 標準的な赤
                else:
                    return 'background-color: #ff4d4d;'  # 濃い赤
            return ''

        # ハイライト適用
        styled_missing_info = missing_info_df.style.applymap(
            gradient_highlight, subset=['欠損率 (%)']
        ).hide(axis='index').set_table_attributes('class="table table-bordered table-hover"')

        return styled_missing_info.to_html()
    
    # 特定の列削除
    def remove_columns(self, columns):
        # 指定した列を削除しデータフレームを更新
        existing_columns = [col for col in columns if col in self.df.columns]
        if not existing_columns:
            raise ValueError('指定した列が見つかりません。:{columns}')
        
        self.df.drop(columns=existing_columns, inplace=True)
        return self
    
    # 欠損値をmean, median, modeで補完
    def fill_missing(self, strategy='mean', columns=None):
        
        if columns in None:
            columns = self.df.select_dtypes(include=['number']).columns.to_list() # 数値列を取得


        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"指定された列'{col}'は存在しません。")
            
            if strategy == 'mean':
                self.df[col].fillna(self.df[col].mean(), inplace=True) # 欠損値を平均値で補完
        
            elif strategy == 'median':
                self.df[col].fillna(self.df[col].median(), inplace=True) # 欠損値を中央値で補完

            elif strategy == 'mode':
                self.df[col].fillna(self.df[col].mode()[0], inplace=True) # 欠損値を最頻値で補完
        
            else:
                raise ValueError(f"strategyには'mean', 'median', 'mode'のいずれかを指定してください。")

        return self
    
    def fill_missing_gradient_boosting(self, target_column, feature_columns, n_estimators=100, learning_rate=0.1, max_depth=3):
        # 欠損値を勾配ブースティングで補完
        # データを欠損値の有無で分割
        if target_column not in self.df.columns:
            raise ValueError(f"指定されたターゲット列'{target_column}'は存在しません。")
        
        missing_features = [col for col in feature_columns if col not in self.df.columns]
        if missing_features:
            raise ValueError(f"指定された特徴量列'{missing_features}'は存在しません。")
        
        train_data = self.df[self.df[target_column].notnull()] # 欠損値がないデータ
        test_data = self.df[self.df[target_column].isnull()] # 欠損値があるデータ

        if train_data.empty or test_data.empty:
            raise ValueError("欠損値のある行または欠損値のない行が存在しません。")
        
        # カテゴリ変数かを判定
        is_categorical = self.df[target_column].dtype == 'object'

        # カテゴリ変数の場合はラベルエンコーディング
        if is_categorical:
            le_target = LabelEncoder()
            train_data[target_column] = le_target.fit_transform(train_data[target_column].astype(str)) # ターゲット列をラベルエンコーディング
            self.encorders[target_column] = le_target # エンコーダを保存

        # 特徴量のエンコーディング
        for col in feature_columns:
           if self.df[col].dtype == 'object':
               le = LabelEncoder()
               train_data[col] = le.fit_transform(train_data[col].astype(str))
               test_data[col] = le.transform(test_data[col].astype(str))
               self.encorders[col] = le

        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_test = test_data[feature_columns]

        # 勾配ブースティングモデルの作成
        model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, radom_state=0)\
            if is_categorical else XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, radom_state=0)
        
        model.fit(X_train, y_train)

        # 欠損値の補完
        predicted_values = model.predict(X_test)
        if is_categorical:
            predicted_values = le_target.inverse_transform(predicted_values)

        self.df.loc[self.df[target_column].isnull(), target_column] = predicted_values

        return self
        

        
        
