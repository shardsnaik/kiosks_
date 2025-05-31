import json

def create_finetune_example(menu_context, dialogue):
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for a food ordering system. Respond using the menu provided."},
            {"role": "user", "content": f"MENU_CONTEXT: {json.dumps(menu_context)}"}
        ] + dialogue
    }

# Menu
menu_context = {
  "food_items": [
    {
      "category": "Milk Shake & Ice Cream",
      "items": [
        { "item_name": "Vennila", "price": "75", "description": "" },
        { "item_name": "Pista", "price": "80", "description": "" }
      ]
    }
  ],
  "offers": ["10% off on order above 499"]
}

# Conversations (short and long)
short_conv = [
    {"role": "user", "content": "Can I get a Vennila milkshake?"},
    {"role": "assistant", "content": "Sure! One Vennila milkshake has been added to your order. Would you like anything else?"},
    {"role": "user", "content": "No, that will be all."},
    {"role": "assistant", "content": "Got it! Your total is ₹75. You can pay via Cash on Delivery. Thank you for your order!"}
]

long_conv = [
    {"role": "user", "content": "Hi, what do you have on the menu?"},
    {"role": "assistant", "content": "We have Milk Shakes like Vennila (₹75) and Pista (₹80). Would you like to try one?"},
    {"role": "user", "content": "Yes, one Vennila and one Pista."},
    {"role": "assistant", "content": "Great! I’ve added Vennila and Pista milkshakes to your cart. That’s ₹155 in total."},
    {"role": "user", "content": "Is there any offer?"},
    {"role": "assistant", "content": "Yes, you get 10% off on orders above ₹499. Your current total is ₹155, so no discount applies yet."},
    {"role": "user", "content": "Okay, go ahead and place the order."},
    {"role": "assistant", "content": "Order confirmed! You can pay on delivery. Enjoy your shakes!"}
]

# Write to JSONL
with open("fine_tune_data.jsonl", "w") as f:
    for conv in [short_conv, long_conv]:
        f.write(json.dumps(create_finetune_example(menu_context, conv)) + "\n")
