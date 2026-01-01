
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QTextEdit, QMessageBox
from PyQt5.QtCore import Qt

from inference import SqlInference

class SQLGeneratorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NL2SQL")
        self.resize(600, 500)
        self.infer = SqlInference()

        self.main_layout = QVBoxLayout()
        
        
        self.col_label = QLabel("Column Count:")
        self.col_input = QLineEdit()
        self.set_col_btn = QPushButton("Set Columns")
        self.set_col_btn.clicked.connect(self.set_columns)
        
        self.add_row_btn = QPushButton("Add Row")
        self.add_row_btn.clicked.connect(self.add_row)
        
        self.first_row = QHBoxLayout() 
        self.first_row.addWidget(self.col_label)
        self.first_row.addWidget(self.col_input)
        self.first_row.addWidget(self.set_col_btn)
        self.first_row.addWidget(self.add_row_btn)
        

        self.table = QTableWidget()
        self.table.setRowCount(2) 
        self.table.setColumnCount(4) 
        self.table.setHorizontalHeaderLabels(["Col 1", "Col 2", "Col 3", "Col 4"])

        self.question_layout = QHBoxLayout()
        self.q_label = QLabel("Question:")
        self.question_input = QLineEdit()
        
        self.question_layout.addWidget(self.q_label)
        self.question_layout.addWidget(self.question_input)

        self.gen_btn = QPushButton("Generate SQL")
        self.gen_btn.setStyleSheet("background-color: #27AE60; color: white;")    
        self.gen_btn.clicked.connect(self.generate_sql)

        self.output_label = QLabel("Generated SQL:")
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setMaximumHeight(50)
        
        self.main_layout.addLayout(self.first_row)
        self.main_layout.addWidget(self.table)
        self.main_layout.addLayout(self.question_layout)
        self.main_layout.addWidget(self.gen_btn)
        self.main_layout.addWidget(self.output_label)
        self.main_layout.addWidget(self.output_display)

        self.setLayout(self.main_layout)

    def set_columns(self):
        txt = self.col_input.text()
        count = int(txt)
        self.table.setColumnCount(count)
        headers = [f"Col {i+1}" for i in range(count)]
        self.table.setHorizontalHeaderLabels(headers)

    def add_row(self):
        current_rows = self.table.rowCount()
        self.table.insertRow(current_rows)

    def get_table_data(self):
        headers = []
        for i in range(self.table.columnCount()):
            item = self.table.horizontalHeaderItem(i)
            if item:
                headers.append(item.text())
            else:
                headers.append(f"col_{i}")
        
        # Get Rows
        rows = []
        for r in range(self.table.rowCount()):
            row_data = []
            is_empty = True
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                text = item.text() if item else ""
                if text.strip():
                    is_empty = False
                row_data.append(text)
            

            if not is_empty:
                rows.append(row_data)
                     
        final_headers = []
        final_rows = []
        
        if self.table.rowCount() > 0:
            for c in range(self.table.columnCount()):
                item = self.table.item(0, c)
                final_headers.append(item.text() if item else f"col_{c}")
        

        for r in range(1, self.table.rowCount()):
            row_data = []
            has_data = False
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                txt = item.text() if item else ""
                if txt.strip():
                    has_data = True
                row_data.append(txt)
            if has_data:
                final_rows.append(row_data)

        return {
            "header": final_headers,
            "rows": final_rows
        }

    def generate_sql(self):
        question = self.question_input.text()
        table_data = self.get_table_data()
        
        generated_sql = self.infer.predict(question, table_data)
        self.output_display.setText(generated_sql)

if __name__ == "__main__":
    app = QApplication([])
    window = SQLGeneratorApp()
    window.show()
    app.exec_()