import smtplib
from email.mime.text import MIMEText
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_alert(detection_type, confidence):
    msg = MIMEText(f'{detection_type} detected with {confidence:.2%} confidence')
    msg['Subject'] = 'Surveillance System Alert'
    msg['From'] = 'alerts@surveillance.com'
    msg['To'] = 'admin@example.com'
    
    try:
        with smtplib.SMTP('localhost', 587) as server:
            server.starttls()
            server.login('username', 'password')
            server.send_message(msg)
        logger.info(f'Alert sent for {detection_type}')
    except Exception as e:
        logger.error(f'Failed to send alert: {e}')