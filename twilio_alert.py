"""
Twilio SMS Alert Script for Underwater Trash Detection

- Sends an SMS alert using Twilio when trash is detected
- Ready to integrate with YOLO-based detection
"""

from twilio.rest import Client

# Twilio credentials (replace with environment variables for production)
ACCOUNT_SID = "US540f6d5937e7a258781fed67a535f7ad"
AUTH_TOKEN = "ae4edad9e061a9ef826cecc49e119fef"
FROM_NUMBER = "+1 415 523 8886"
TO_NUMBER = "+91 7002168639"

# Initialize Twilio client
client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_alert(message: str):
    """
    Sends an SMS alert with the given message using Twilio.
    Args:
        message (str): The alert message to send.
    Returns:
        The message SID if sent successfully.
    """
    try:
        sms = client.messages.create(
            body=message,
            from_=FROM_NUMBER,
            to=TO_NUMBER
        )
        print(f"✅ Alert sent! Message SID: {sms.sid}")
        return sms.sid
    except Exception as e:
        print(f"❌ Failed to send alert: {e}")
        return None

# Simulate detection event (for testing)
if __name__ == "__main__":
    # Example: Call this function when YOLO detects trash
    send_alert("🚨 Trash detected underwater at location (12.34, 56.78)") 