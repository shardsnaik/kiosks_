from PyQt6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout,
    QWidget, QHBoxLayout, QTextEdit, QListWidget, QListWidgetItem,
    QScrollArea, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap

from main.pipelines.menu_maneger_pipeline import MenuManagerPipeline

obj = MenuManagerPipeline()


class MenuItemWidget(QFrame):
    def __init__(self, item_data, add_to_cart_callback):
        super().__init__()
        
        self.item_data = item_data
        
        # Set frame style
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(1)
        self.setStyleSheet("background-color: #f8f8f8; border-radius: 8px; margin: 5px;")
        
        # Layout
        layout = QVBoxLayout()
        
        # Item name
        name_label = QLabel(item_data["item_name"])
        name_font = QFont()
        name_font.setBold(True)
        name_font.setPointSize(12)
        name_label.setFont(name_font)
        layout.addWidget(name_label)
        
        # Price
        price_label = QLabel(f"₹{item_data['price']}")
        price_font = QFont()
        price_font.setPointSize(10)
        price_label.setFont(price_font)
        price_label.setStyleSheet("color: #d35400;")
        layout.addWidget(price_label)
        
        # Description
        if item_data.get("description"):
            desc_label = QLabel(item_data["description"])
            desc_label.setWordWrap(True)
            desc_font = QFont()
            desc_font.setPointSize(9)
            desc_label.setFont(desc_font)
            desc_label.setStyleSheet("color: #7f8c8d;")
            layout.addWidget(desc_label)
        
        # Add to cart button
        add_button = QPushButton("Add to Cart")
        add_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        add_button.clicked.connect(lambda: add_to_cart_callback(item_data))
        layout.addWidget(add_button)
        
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Sample menu data (this would come from your JSON)
        self.menu_data = obj.run_pipeline()
        
        self.cart_items = []  # To store cart items with quantity and price
        self.total_price = 0

        self.setWindowTitle("The Kiosks")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color: #ecf0f1;")

        # Main container
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # ---------------------------
        # Left Panel (Menu Area)
        # ---------------------------
        left_layout = QVBoxLayout()

        # Welcome header
        welcome_label = QLabel("Welcome to The KIOSKS Palace")
        font = welcome_label.font()
        font.setPointSize(24)
        font.setBold(True)
        welcome_label.setFont(font)
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        welcome_label.setStyleSheet("color: #34495e; margin-bottom: 15px; padding: 10px;")
        left_layout.addWidget(welcome_label)
        
        # Menu label
        menu_label = QLabel("📋 Our Menu")
        menu_font = menu_label.font()
        menu_font.setPointSize(18)
        menu_font.setBold(True)
        menu_label.setFont(menu_font)
        menu_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        menu_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        left_layout.addWidget(menu_label)

        # Scrollable menu area
        menu_scroll = QScrollArea()
        menu_scroll.setWidgetResizable(True)
        menu_scroll.setStyleSheet("border: none;")
        
        menu_container = QWidget()
        self.menu_layout = QVBoxLayout(menu_container)
        
        # Add menu items
        self.populate_menu()
        
        menu_scroll.setWidget(menu_container)
        left_layout.addWidget(menu_scroll)

        # ---------------------------
        # Right Panel (Chat + Cart)
        # ---------------------------
        right_layout = QVBoxLayout()
        
        # Chat area
        chat_label = QLabel("💬 Chat with Staff")
        chat_label.setFont(menu_font)
        chat_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chat_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        right_layout.addWidget(chat_label)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("background-color: white; border-radius: 8px; padding: 10px;")
        self.chat_display.append("Staff: Hello! Welcome to Taj Mahal Palace. What would you like to order today?")
        right_layout.addWidget(self.chat_display)
       
        # Cart display
        cart_label = QLabel("🛒 Your Order")
        cart_label.setFont(menu_font)
        cart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cart_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        right_layout.addWidget(cart_label)

        self.cart_list = QListWidget()
        self.cart_list.setStyleSheet("background-color: white; border-radius: 8px;")
        right_layout.addWidget(self.cart_list)
        
        # Total price display
        self.total_price_label = QLabel("Total: ₹0")
        total_font = self.total_price_label.font()
        total_font.setPointSize(16)
        total_font.setBold(True)
        self.total_price_label.setFont(total_font)
        self.total_price_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.total_price_label.setStyleSheet("color: #27ae60; margin: 10px;")
        right_layout.addWidget(self.total_price_label)
        
        # Checkout button
        checkout_button = QPushButton("Proceed to Checkout")
        checkout_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border-radius: 6px;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        checkout_button.clicked.connect(self.checkout)
        right_layout.addWidget(checkout_button)

        # Combine layouts
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 2)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def populate_menu(self):
        # Clear existing menu items
        while self.menu_layout.count():
            child = self.menu_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add menu items from JSON data
        for item_data in self.menu_data:
            item_widget = MenuItemWidget(item_data, self.add_to_cart)
            self.menu_layout.addWidget(item_widget)
        
        # Add stretch to push items to the top
        self.menu_layout.addStretch()

    def add_to_cart(self, item_data):
        # Add item to cart
        self.cart_items.append(item_data)
        
        # Update chat
        self.chat_display.append(f"You: I'll have one {item_data['item_name']}, please.")
        self.chat_display.append(f"Staff: One {item_data['item_name']} added to your order!")
        
        # Update cart list
        self.update_cart_display()
        
        # Scroll chat to bottom
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def update_cart_display(self):
        # Clear cart
        self.cart_list.clear()
        
        # Count item occurrences
        item_counts = {}
        for item in self.cart_items:
            name = item["item_name"]
            if name in item_counts:
                item_counts[name]["count"] += 1
            else:
                item_counts[name] = {"count": 1, "price": int(item["price"])}
        
        # Calculate total
        self.total_price = 0
        
        # Add items to cart list
        for name, data in item_counts.items():
            count = data["count"]
            price = data["price"]
            subtotal = count * price
            self.total_price += subtotal
            
            self.cart_list.addItem(f"{name} x{count} - ₹{subtotal}")
        
        # Update total price display
        self.total_price_label.setText(f"Total: ₹{self.total_price}")

    def checkout(self):
        if not self.cart_items:
            self.chat_display.append("Staff: Your cart is empty. Please add items before checkout.")
            return
            
        self.chat_display.append("Staff: Thank you for your order! Your food will be ready shortly.")
        
        # In a real app, you would process payment, generate order numbers, etc.
        self.cart_items = []
        self.update_cart_display()


# Run App
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()