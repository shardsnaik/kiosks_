from PyQt6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout,
    QWidget, QHBoxLayout, QTextEdit, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My Kiosk App")
        self.setGeometry(100, 100, 1000, 600)

        # Main container
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # ---------------------------
        # Left Panel (Chat Area)
        # ---------------------------
        left_layout = QVBoxLayout()

        welcome_label = QLabel("Welcome to the Kiosk App!")
        font = welcome_label.font()
        font.setPointSize(20)
        welcome_label.setFont(font)
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        welcome_label.setStyleSheet("background-color: lightblue; padding: 10px;")
        menu_label = QLabel("📋 Menu")
        menu_label.setFont(font)
        menu_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(menu_label)

        # Menu Buttons
        self.menu_items = ["Burger", "Fries", "Soda", "Pizza", "Salad"]
        for item in self.menu_items:
            btn = QPushButton(f"Add {item}")
            btn.clicked.connect(lambda checked, i=item: self.add_to_cart(i))
            left_layout.addWidget(btn)


        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.append("Cashier: Hello! What would you like to order today?")

        left_layout.addWidget(welcome_label)
        left_layout.addWidget(self.chat_display)

        # ---------------------------
        # Right Panel (Menu + Cart)
        # ---------------------------
        right_layout = QVBoxLayout()

       
        # Cart display
        cart_label = QLabel("🛒 Cart")
        cart_label.setFont(font)
        cart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(cart_label)

        self.cart_list = QListWidget()
        right_layout.addWidget(self.cart_list)

        # Combine layouts
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def add_to_cart(self, item_name):
        self.chat_display.append(f"Customer: I'll take one {item_name}, please.")
        self.chat_display.append(f"Cashier: One {item_name}, got it!")
        self.cart_list.addItem(QListWidgetItem(item_name))


# Run App
app = QApplication([])
window = MainWindow()
window.show()
app.exec()
