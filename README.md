# Smart Glove

Most accessibility devices for the visually impaired are text to speech based which is not ideal for people who may be both visually and auditorily impaired (such as the elderly). Our solution provides an opportunity for such users to make a braille output and adapts in situations when person uses it outside (guesses on which text - i.e. signboard the person looks and translates only it).

This project utilizes **Optical Character Recognition (OCR)** to extract text from video frames in real-time, using **OpenCV** and **Tesseract OCR**. It implements **optical flow** to select relevant points-of-interest and **contextual grouping** to organize detected texts based on **spatial proximity**. This allows end-users to navigate their environment more intuitively and effectively, enhancing their independence and confidence. 

Project is split into **client** and **server** parts. **Client** must be located on the glove, while **server** may be placed either on the glove as well, either on mobile phone or remote server.

This project was made in less then 24h for MakeUofT hackathon in January 2024. I want to express my gratitude to **Alexander Apostolu**, **Nearhos Hatzinikolaou** and **George Dobrik**, they wrote code for OCR and hardware, also helped in assembling the glove and software integration.

Team photo with the prototype:
![Team Photo with the prototype](https://github.com/user-attachments/assets/8007fc1b-3432-4130-8395-470d2bcaa277)
