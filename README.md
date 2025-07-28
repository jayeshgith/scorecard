# Kabaddi Scorecard Automation

This is a project I built to automate Kabaddi scoring using hand gesture detection. It uses a webcam to track the umpairs' gestures in real-time, and based on the gesture shown, the score is updated automatically.

I used Python with Flask for the backend, and HTML, CSS, and JavaScript for the frontend. MediaPipe is used to detect the hand and elbow positions from the video stream.

What it does

- Tracks hand gestures like left1 to left5, right1 to right5, and bonus gestures using MediaPipe.
- Adds points automatically based on the detected gesture.
- Has a live video feed on the frontend to show what’s being captured.
- Includes a timer for each round.
- Allows match results to be saved in the browser using localStorage.
- Has a login system and a page to view saved match data.

 Technologies Used

- Frontend: HTML, CSS, JavaScript
- Backend: Python (Flask)
- Computer Vision: OpenCV and MediaPipe


How to Run

1. Clone the repository and go into the folder.
2. https://github.com/jayeshgith/scorecard

