from flask import Flask, request, render_template, redirect, url_for, send_file
from models import db, UploadedFile
from csv_processor import CSVProcessor
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///files.db'
db.init_app(app)

with app.app_context():
    db.create_all()

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # 同じファイル名が既に存在する場合は削除
            existing_file = UploadedFile.query.filter_by(filename=file.filename).first()
            if existing_file:
                os.remove(existing_file.filepath)
                db.session.delete(existing_file)
                db.session.commit()

            # 新しいファイルをDBに保存
            uploaded_file = UploadedFile(filename=file.filename, filepath=filepath)
            db.session.add(uploaded_file)
            db.session.commit()
            return redirect(url_for('display_file', file_id=uploaded_file.id))

    files = UploadedFile.query.order_by(UploadedFile.created_at.desc()).all()
    return render_template('index.html', files=files)

@app.route('/file/<int:file_id>')
def display_file(file_id):
    file_record = UploadedFile.query.get_or_404(file_id)
    processor = CSVProcessor(file_record.filepath)

    table = processor.get_preview()
    basic_info = processor.get_basic_info()
    missing_info = processor.highlight_missing_info(threshold=10)
    all_columns = processor.df.columns.tolist()
    files = UploadedFile.query.order_by(UploadedFile.created_at.desc()).all()

    return render_template('index.html', table=table, basic_info=basic_info, missing_info=missing_info,
                           filename=file_record.filename, all_columns=all_columns, files=files, current_file_id=file_id)

@app.route('/remove_columns/<int:file_id>', methods=['POST'])
def remove_columns(file_id):
    file_record = UploadedFile.query.get_or_404(file_id)
    processor = CSVProcessor(file_record.filepath)

    columns_to_remove = request.form.getlist('columns_to_remove')

    if not columns_to_remove:
        return redirect(url_for('display_file', file_id=file_id, error="削除する列を選択してください。"))

    try:
        processor.remove_columns(columns_to_remove)
        processor.df.to_csv(file_record.filepath, index=False)
        return redirect(url_for('display_file', file_id=file_id))

    except Exception as e:
        return redirect(url_for('display_file', file_id=file_id, error=str(e)))

@app.route('/fill_missing/<int:file_id>', methods=['POST'])
def fill_missing(file_id):
    file_record = UploadedFile.query.get_or_404(file_id)
    processor = CSVProcessor(file_record.filepath)

    columns_to_fill = request.form.getlist('columns_to_fill')
    strategy = request.form.get('strategy')

    if not columns_to_fill:
        return redirect(url_for('display_file', file_id=file_id, error="補完する列を選択してください。"))

    try:
        processor.fill_missing(strategy=strategy, columns=columns_to_fill)
        processor.df.to_csv(file_record.filepath, index=False)
        return redirect(url_for('display_file', file_id=file_id))

    except Exception as e:
        return redirect(url_for('display_file', file_id=file_id, error=str(e)))

@app.route('/fill_missing_gb/<int:file_id>', methods=['POST'])
def fill_missing_gb(file_id):
    file_record = UploadedFile.query.get_or_404(file_id)
    processor = CSVProcessor(file_record.filepath)

    target_column = request.form.get('target_column')
    feature_columns = request.form.getlist('feature_columns')
    n_estimators = int(request.form.get('n_estimators', 100))
    learning_rate = float(request.form.get('learning_rate', 0.1))
    max_depth = int(request.form.get('max_depth', 3))

    if not target_column or not feature_columns:
        return redirect(url_for('display_file', file_id=file_id, error="補完するターゲット列と特徴量を選択してください。"))

    try:
        processor.fill_missing_gradient_boosting(target_column, feature_columns, n_estimators, learning_rate, max_depth)
        processor.df.to_csv(file_record.filepath, index=False)
        return redirect(url_for('display_file', file_id=file_id))

    except Exception as e:
        return redirect(url_for('display_file', file_id=file_id, error=str(e)))

@app.route('/delete_file/<int:file_id>', methods=['POST'])
def delete_file(file_id):
    file_record = UploadedFile.query.get_or_404(file_id)

    # ファイルを削除
    try:
        os.remove(file_record.filepath)
    except FileNotFoundError:
        pass

    # データベースからも削除
    db.session.delete(file_record)
    db.session.commit()

    return redirect(url_for('upload_file'))

@app.route('/download/<int:file_id>')
def download_file(file_id):
    file_record = UploadedFile.query.get_or_404(file_id)
    return send_file(file_record.filepath, as_attachment=True, download_name=file_record.filename)

if __name__ == '__main__':
    app.run(debug=True)

