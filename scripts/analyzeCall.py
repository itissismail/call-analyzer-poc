import json

def analyze_call(json_path):
    # Load JSON file
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Extract segments (modify key if your structure differs)
    segments = data.get("segments", [])
    if not segments and isinstance(data, list):
        segments = data
    
    # Sort by start time
    segments = sorted(segments, key=lambda x: x.get("start", 0))
    
    # Prepare conversation text
    conversation_text = []
    for seg in segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        if text:
            conversation_text.append(f"[{start:.2f}-{end:.2f}] {text}")
    
    # Detect topics (simple keyword search)
    print("\n--- Topics ---")
    topics = []
    for seg in segments:
        t = seg.get("text", "").lower()
        print(f"Topic: {t}")
        if "bill" in t or "payment" in t:
            topics.append("Billing/Payment")
        elif "account" in t:
            topics.append("Account Issue")
        elif "service" in t or "support" in t:
            topics.append("Service/Support")
    topics = list(set(topics))  # unique
    
    print("\n--- End Topics ---")
    # Heuristic scoring
    greeting_score = 8 if any("hello" in seg.get("text", "").lower() or "welcome" in seg.get("text", "").lower()
                              for seg in segments) else 5
    handling_score = 8 if any("help" in seg.get("text", "").lower() or "check" in seg.get("text", "").lower()
                              for seg in segments) else 6
    resolution_score = 9 if any("thank" in seg.get("text", "").lower() or "resolved" in seg.get("text", "").lower()
                                for seg in segments) else 6
    overall_score = round((greeting_score + handling_score + resolution_score) / 3, 1)
    
    # Print results
    print("\n--- Full Conversation ---")
    print("\n".join(conversation_text))
    
    print("\n--- Detected Topics ---")
    print(", ".join(topics) if topics else "No clear topics detected.")
    
    print("\n--- Call Quality Scores (out of 10) ---")
    print(f"Greeting: {greeting_score}")
    print(f"Handling: {handling_score}")
    print(f"Resolution: {resolution_score}")
    print(f"Overall: {overall_score}")

# Usage
analyze_call("./results/sample.json")  # Replace with your file path
