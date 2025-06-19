import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "Urgent: Your account has been compromised! Click here to secure your funds immediately!",

"Congratulations! You've won a $5,000 cash prize! To claim it, just provide your bank details.",

"Hurry, you’ve been pre-approved for a $10,000 loan! Just pay a small upfront fee to receive it.",

"You’ve won a luxury car! Confirm your shipping address to receive it now.",

"Your account is at risk! Click here to verify your identity before it’s locked.",

"You’ve been selected for a $1,000 prize! Pay a small fee to claim your money.",

"Urgent: We detected suspicious activity on your account. Click here to prevent your account from being suspended.",

"Your Amazon account has been locked. Click here to verify your details and regain access.",

"You’ve been selected for a $500 cash reward! Confirm your details to receive it.",

"Congratulations! You’ve won a free vacation to Paris! Just pay for taxes to confirm your booking.",

"We need to verify your account. Click here to prevent your account from being frozen.",

"Your bank has issued a refund of $2,000! Confirm your account details to receive it.",

"You’ve won an exclusive iPhone 12! Confirm your shipping details to have it sent to you today.",

"Click here to unlock your $1,000 prize! Only available for the next 24 hours!",

"Your loan application has been approved! Just pay a small processing fee to receive the funds.",

"Your account has been temporarily suspended. Click here to verify your information and restore access.",

"You’ve won $10,000! Confirm your identity and pay the processing fee to claim it.",

"Urgent: Your PayPal account has been flagged! Click here to confirm your account details and avoid suspension.",

"Click here to unlock your free gift card worth $500! Available for a limited time only!",

"Congratulations! You’ve won a brand-new car! Just pay for delivery and handling to claim it.",

"Hurry, this is your final chance to claim your $500 reward! Confirm your details now.",

"Urgent: Your bank account has been compromised. Click here to lock your funds and prevent unauthorized access.",

"You’ve been selected for an exclusive reward! Click here to confirm your eligibility.",

"Your application for a loan has been approved! Just pay a small fee to complete the process.",

"Urgent: Your account has been hacked! Click here to reset your password and secure your account.",

"Congratulations! You’ve won a free vacation to Bali! Just pay the small booking fee to confirm your trip.",

"Click here to redeem your free gift card worth $1,000! Limited-time offer!",

"You’ve been pre-approved for a credit card with a $5,000 limit! Pay the activation fee to receive it.",

"Your prize is waiting for you! Confirm your bank details to claim your reward.",

"Urgent: You’ve received a refund of $500! Click here to confirm your details to receive the refund.",

"Your payment has been flagged! Click here to resolve the issue before it’s too late.",

"You’ve won a luxury watch! Confirm your shipping details to have it sent to you immediately.",

"Congratulations! You’ve been selected for an exclusive $1,000 gift card! Click here to claim it.",

"Urgent: Your credit card has been suspended. Click here to verify your account details and lift the suspension.",

"You’ve won a shopping spree worth $2,000! Just provide your bank details to claim your prize.",

"Congratulations! You’ve been selected for a free iPad. Just confirm your shipping address to receive it.",

"Urgent: Your bank account has been frozen. Click here to verify your details and restore access.",

"Click here to redeem your free $500 gift card! Hurry, offer ends soon!",

"You’ve won a $10,000 prize! Just pay a small processing fee to claim it.",

"Urgent: You’ve been selected for a free credit check. Click here to confirm your details and secure your identity.",

"You’ve won a free iPhone! Confirm your shipping address to have it delivered to you today.",

"Click here to claim your $500 reward! Just confirm your account details to receive it.",

"Your account has been flagged for suspicious activity. Click here to verify your details and prevent suspension.",

"Congratulations! You’ve won a brand-new laptop! Confirm your address to receive it.",

"You’ve won $1,000! Click here to confirm your bank details to receive it.",

"Urgent: Your PayPal account has been compromised. Click here to verify your details and secure your account.",

"Congratulations! You’ve won a free cruise! Just pay for taxes to confirm your booking.",

"Click here to claim your $2,000 prize! Only available for the next 48 hours!",

"Urgent: Your account needs verification. Click here to confirm your details and avoid suspension.",

"You’ve been selected to receive a free laptop! Just confirm your address and pay the shipping fee.",

    "Urgent: Your bank account has been compromised, click here to secure it now!",

"You’ve been selected for a $2,000 reward! Please send a small processing fee to claim it.",

"Your prize is waiting! To claim your $500 gift card, just provide your details.",

"Hurry, this offer expires soon! Click here to get your free laptop.",

"Congratulations! You’ve won a lottery worth $10,000, confirm your information to claim!",

"Your account has been suspended. Please click here to restore access immediately!",

"Exclusive offer: Receive $500 just by filling out a quick survey, act fast!",

"You qualify for a $1,000 cash reward! Pay a small fee to claim it.",

"Dear user, your account will be permanently closed unless you verify your details today!",

"This is your last chance! Confirm your details to claim your free iPhone 12.",

"You’ve been approved for a $10,000 loan, simply pay a small application fee to receive it.",

"Your Amazon account has been flagged. Click here to secure your account now.",

"Congratulations! You’ve been chosen to receive $5,000 in prize money, pay a small tax fee to claim it.",

"You’re eligible for a free vacation, pay a small booking fee to confirm your spot!",

"Click here to activate your free gift card worth $1,000, only available today!",

"Immediate action required! Your credit card has been compromised, click here to secure your account.",

"Great news! You've won a free iPhone! Confirm your address to have it shipped today.",

"You’ve been pre-approved for a loan, just pay the processing fee to finalize the application.",

"You’re eligible for a $300 reward, just confirm your bank details to receive it.",

"Congratulations, your entry has won a free shopping spree! Confirm your payment details to claim your prize.",

"You’ve received a $100 gift card, just pay a small activation fee to use it.",

"Click here to get your free trial of premium software, no credit card required!",

"You’ve been selected to receive $2,000! Confirm your identity to claim it.",

"Urgent: Your account has been temporarily suspended. Click here to regain access.",

"Congratulations! You’re entitled to a free $500 gift card, simply pay for shipping.",

"We have a special offer for you! Get $500 for completing a simple survey.",

"You have a limited time to claim your free electronics! Confirm your shipping info now.",

"Click here to claim your prize money, $2,000 waiting for you!",

"Congratulations! You've won a $1000 gift card, verify your details to receive it.",

"Your bank account has been frozen due to suspicious activity, click here to unfreeze it.",

"You're eligible for a $5,000 grant, pay a small fee to receive it.",

"Urgent: Click here to update your account information to avoid being locked out.",

"Special offer: Get a free Amazon gift card by signing up for our service today!",

"You’ve won a lottery prize worth $500, confirm your details to claim it now.",

"You’ve received a bonus payment of $1,000! Verify your account to unlock it.",

"Your account has been flagged for unusual activity. Click here to secure it now.",

"You’ve been pre-approved for a credit card with a $10,000 limit, just pay a small processing fee.",

"Hurry! Only 24 hours left to claim your free gift card worth $1,000.",

"You’ve won a free vacation! Confirm your details to book your trip today.",

"This is a one-time offer! Get your free iPhone 13 by completing a short survey.",

"Congratulations, you’ve won a free luxury watch! Pay the shipping fee to receive it.",

"Your Amazon account is at risk! Click here to verify your information to prevent suspension.",

"You’ve won a free trip to Paris! Click here to claim your free airfare.",

"We need to verify your account before we can release your $500 prize money.",

"You’ve been selected for a $200 reward! Just confirm your bank details to claim it.",

"Immediate action required! Your social media account is at risk, click here to secure it.",

"Urgent: Your subscription has been renewed, click here to cancel and get a refund.",

"Congratulations! You’ve won a brand-new laptop! Confirm your shipping details to receive it.",

"Your loan application has been approved! Pay a small fee to receive your funds.",

"We’re offering a special bonus! Get a $1,000 reward for completing this form.",

"You’ve been selected to win a $500 shopping spree! Just confirm your email address to claim it.",

"Hurry! You’ve been pre-approved for a $10,000 loan. Pay a small upfront fee to finalize your application.",

"We’ve detected suspicious activity on your account. Please confirm your identity immediately.",

"You’ve won a luxury car! Just pay the shipping fee to have it delivered to your doorstep.",

"Urgent! You have $2,000 waiting for you. Please confirm your details to receive it.",

"Click here to claim your free gift card worth $500, only available for the next 24 hours!",

"You’ve been approved for a credit card with a $10,000 limit! Just pay a small fee to receive it.",

"We need to verify your payment method to process your $1,000 prize claim.",

"You’ve won a free vacation! Just pay the booking fee to confirm your stay.",

"Act now! You can get a $1,000 gift card by answering a few simple questions.",

"You’re eligible to receive a $500 reward, confirm your details to receive it now.",

"This is a limited-time offer! Get a free iPhone just by signing up today.",

"Congratulations! You’ve been selected to receive a $1,000 reward, pay a small activation fee to claim it.",

"Click here to claim your $500 gift card! All you need to do is confirm your email address.",

"You’ve won a free holiday getaway! Just pay a small processing fee to finalize your booking.",

"Your account has been temporarily suspended. Click here to unlock it by verifying your details.",

"Congratulations! You’ve won a $500 gift card, pay a small handling fee to receive it.",

"You’ve been selected for a special offer! Receive $1,000 by confirming your details.",

"Your bank account will be locked unless you verify your details within 24 hours.",

"You’ve won a $2,000 prize! Simply pay a small processing fee to receive your reward.",

"Act quickly! You’ve been pre-approved for a $1,000 loan. Pay the upfront fee to claim it.",

"Click here to secure your $100 gift card by filling out a quick survey!",

"You’ve been selected for a cash reward worth $500, confirm your account details to claim it.",

"Congratulations! You’ve won an all-expenses-paid trip! Pay a small fee to finalize your booking.",

"We’ve added a bonus to your account! Confirm your information to claim your reward.",

"Your account has been locked. Please verify your information to unlock it immediately.",

"You’ve been pre-approved for a $10,000 loan! Pay the processing fee to receive your funds.",

"Get your free gift card now by clicking here and filling out a simple form!",

"Congratulations! You’ve won a free iPad. Confirm your shipping details to receive it.",

"You’ve been selected for a special promotion! Receive $500 by completing a short form.",

"We’ve detected fraudulent activity on your account. Click here to secure it immediately.",

"You’ve won a vacation for two! Pay a small fee to confirm your reservation.",

"Congratulations! You’re eligible to claim $1,000, confirm your account details to claim your prize.",

"You’re entitled to receive $500. Confirm your details to receive the payment.",

"Your email address has been selected for a $1,000 reward. Click here to confirm.",

"Congratulations! You’ve been chosen to receive $500, confirm your details to get it now.",

"This is a one-time offer! Get a free $1,000 Amazon gift card today!",

"Urgent! Your account needs immediate verification to avoid being locked out.",

"You’ve been selected for a free gift worth $500, just confirm your shipping address.",

"Click here to activate your free gift card worth $200, available for a limited time only!",

"You’ve won a free vacation! Confirm your details now to book your trip.",

"You’ve been selected for a $1,000 prize, just confirm your payment details to receive it.",

"Immediate action required! Confirm your details to unlock your $500 cash reward.",

"You’ve won a free gadget! Just pay for shipping and handling to receive it.",

"Congratulations! You’ve been chosen for a $2,000 prize, confirm your identity to claim it.",

"Click here to secure your free $500 gift card, available only for the next 48 hours!",

"Your account has been flagged for suspicious activity, click here to verify your information.",

"You’ve won a $500 reward! Confirm your details to receive your prize.",

"You’ve been pre-approved for a loan of up to $5,000, just pay the processing fee to receive it.",

"Congratulations! You’ve been chosen to receive $1,000 in prize money, confirm your details to claim it.",
    "Congratulations! You've been selected to win a $1,000 Amazon gift card. Click here to claim it now!",

"URGENT: Your account has been compromised. Please log in immediately to secure it.",

"Exclusive offer: Buy 1, get 10 free! Hurry, this offer expires in 24 hours!",

"You've won a free iPhone! Just pay shipping and handling to claim your prize.",

"Final notice: Your bank account will be suspended unless you update your details immediately.",

"Limited-time offer: Get $500 just by sharing your PayPal email address!",

"Hey, I'm a recruiter from a high-paying job company. Are you looking for remote work?",

"Your prize is ready to claim! Send a small fee for processing, and it’s all yours.",

"Dear customer, your account has been locked due to suspicious activity. Please click here to reset your password.",

"This is your last chance to claim your $1,000 prize. Don’t miss out!",

"You’ve been chosen to receive an all-expenses-paid vacation! Just pay the registration fee now.",

"Congratulations! You are the grand prize winner of our lottery. Please send us a processing fee to claim it.",

"Hi, we noticed some unusual activity on your credit card. Verify your identity to avoid account suspension.",

"Offer only for today: Pay $50 to receive $500 in return! This is a limited-time opportunity.",

"We’ve added $100 to your bank account as a reward! Just verify your information to release the funds.",

"You’ve been selected for an exclusive loan offer! Fast approval with no credit check required.",

"You’ve won a free vacation to Paris! Just cover the taxes and fees to confirm.",

"This is not a scam! Transfer $200 now and receive $2,000 in the next 48 hours!",

"URGENT: Your Netflix subscription is about to expire. Update your payment details now to avoid interruption.",

"This is a one-time offer! Click here to get a free $500 gift card to Walmart.",

"Hello! You’ve been randomly selected for a free car! Just pay the shipping charges.",

"We are reaching out to you because you have outstanding debt. Pay within 48 hours to avoid legal action.",

"We’ve detected a security breach in your bank account. Click here to secure it immediately.",

"Last chance: Claim your $1,000 prize by entering your payment details below.",

"Special investment opportunity! Double your money within 30 days. Limited spaces available.",

"Your PayPal account has been suspended. Please click here to reactivate it.",

"Act now! You can receive a $500 gift card if you fill out this short survey.",

"We need your credit card details to process a transaction on your account.",

"Exclusive deal for today only! Get 3 months of free Amazon Prime when you sign up now!",

"You've been selected for a $10,000 grant! Send a small fee to claim your funds.",

"Win an iPad in 3 easy steps! Step 1: Pay for shipping. Step 2: Wait for delivery.",

"URGENT: Your account has been flagged. Please verify your identity by clicking here.",

"You’ve been pre-approved for a $50,000 loan with 0% interest! Click here to apply now.",

"You have 24 hours to claim your free iPad. Act fast to secure it!",

"Your Amazon account has been temporarily suspended due to suspicious activity. Click to reactivate.",

"Hurry! Sign up for this free program and get $100 instantly deposited into your account!",

"We’ve identified a problem with your account. Please send $100 to fix it.",

"This is not a drill! Earn $1,000 a week by working from home. Apply now!",

"We are offering you a free trial of our software, but you need to pay a small upfront fee for shipping.",

"Your account has been compromised! Please reset your password and update your information now.",

"Congratulations, you’ve won the jackpot! To claim your $1,000, send a small processing fee.",

"Hurry! Get a free consultation and win $500 cash if you sign up today.",

"Your loan application has been approved. Send a fee of $200 to release the funds.",

"This is a reminder that your PayPal account has been compromised. Please click here to secure it.",

"We’re offering you a once-in-a-lifetime investment opportunity. Double your money with minimal risk!",

"Congratulations! You've been selected to receive a $1,000 rebate. Please pay the processing fee to receive it.",

"Click here to claim your $1,000 prize. All we need is a payment for taxes.",

"You’ve been selected for an exclusive VIP membership! Pay now to join the program.",

"You’ve been selected for a $1,500 loan with no interest for 12 months. Apply now!",

"URGENT: We’ve detected fraudulent activity on your account. Please verify your details immediately.",

"Hurry! Free iPad giveaway if you enter your payment details now.",

"Get a guaranteed $500 payout by signing up for this investment program!",

"Congratulations, you've just been chosen for a one-time giveaway. Pay shipping fees to claim your prize.",

"Click here to redeem your $500 reward, exclusive to new customers!",

"You’ve been selected for a special offer on car insurance. Save big by signing up today!",

"Limited-time offer: Receive a free consultation worth $500 by signing up now.",

"Your phone number has won $1,000! Simply reply to claim it.",

"IMPORTANT: Your email account is at risk. Click here to secure it before it’s too late!",

"We’ve approved your loan! Just send a processing fee to finalize the transfer.",

"You are pre-qualified for a $10,000 grant. To complete your application, click here.",

"URGENT: Action required! Your social security number has been compromised. Click to protect it.",

"Big news! You've been chosen for a $500 reward. Confirm your details to claim it.",

"Don’t miss out! Exclusive $1,000 prize for responding to this email.",

"Congratulations! You’ve been picked for a new car giveaway. Just pay a small fee for delivery.",

"You've been randomly selected for a free luxury vacation! Pay taxes and fees to confirm.",

"Your account will be suspended unless you update your information now. Click here to proceed.",

"Here’s your opportunity to earn $2,000 a week working from home. Apply now!",

"Only today: $500 free bonus if you open an account now!",

"You’ve won the lottery! Pay a small fee to receive your $1,000,000 prize.",

"Final notice! Your account is about to be suspended unless you update your payment details.",

"Limited time: Free gift cards worth $100! Click here to claim yours.",

"Congratulations, you’ve been selected to receive a vacation! Just pay the taxes and you’re good to go.",

"You’ve earned a reward for completing a quick survey! Just pay the small shipping fee to receive it.",

"Exclusive offer: Win $1,000 by signing up for this service right now!",

"Action required: Confirm your bank details to avoid account suspension.",

"Hurry! Sign up now for a chance to win a brand-new laptop, no strings attached.",

"Special deal: Pay $50 for an all-expenses-paid trip to Hawaii!",

"Your $2,000 prize is waiting! Simply confirm your details to claim it.",

"Get paid to work from home today! Apply to start earning $1,000 per week.",

"URGENT: Your bank account has been flagged for suspicious activity. Click here to fix it.",

"We've added $200 to your account! Just verify your payment method to claim it.",

"Congratulations! You've won a free cruise! Just pay taxes to book your cabin.",

"This is a one-time opportunity to get a $500 bonus. Sign up now to claim it.",

"You've been approved for a fast loan. Just send a small fee to release the funds.",

"FINAL WARNING: Your account has been locked due to unusual activity. Verify your identity now!",

"Act fast! This is your last chance to claim your free gift card worth $1,000.",

"We’ve added a bonus to your account! Please confirm your details to release the funds.",

"You’re eligible for a $1,000 payout! Just confirm your bank details to receive it.",

"Hurry! Receive $200 by signing up for this amazing offer right now!",

"Congratulations, you’ve won a $500 shopping spree! Confirm your email to claim your prize.",

"Special offer: Send $200 and receive a free iPhone 14!",

"You've won a contest! To claim your $500 reward, just send a small fee.",

"Get a guaranteed $1,000 payout by signing up for this opportunity today!",

"Your bank account has been frozen due to suspicious activity. Click here to restore access.",

"Urgent: Your account will be closed unless you provide your login details now.",

"You’ve been selected to receive a $10,000 grant. Pay a small fee to claim your funds.",

"You’ve received a special reward! Pay a fee to activate your free gift.",

"Congratulations! You’ve won a free vacation. Pay the processing fee to book it.",

"You’ve been pre-approved for a loan. To receive it, please send a processing fee.",

"Special promotion: Receive $100 in Amazon gift cards by completing this quick survey!",
    "I hope you're having a great day!",
"Let's meet at the cafe this afternoon.",
"Can you send me the details of the project by tomorrow?",
"Don't forget to take your umbrella; it's supposed to rain later.",
"I finished reading that book you recommended. It was fantastic!",
"I'll be attending the meeting at 3 PM today.",
"We need to schedule a team call to discuss the new project.",
"Can you help me with the grocery list for the weekend party?",
"The movie starts at 7 PM. Do you want to come along?",
"I’ve booked the tickets for the concert this weekend.",
"I’m planning to visit my parents this Saturday.",
"Do you have any plans for the upcoming holidays?",
"I will be sending over the document by noon tomorrow.",
"This restaurant has great reviews, we should try it sometime.",
"I'm looking forward to catching up with you this weekend.",
"I need to send the email before the end of the day.",
"Let's grab lunch together at the new Italian place.",
"I need your feedback on the presentation slides.",
"Can you remind me to call Sarah later?",
"Please confirm if you're available for the meeting on Monday.",
"The weather looks great today, perfect for a walk.",
"I’m so excited for the concert this weekend!",
"Let’s plan a weekend getaway for the summer.",
"I have a doctor's appointment later today.",
"My friend is getting married next month, and I can’t wait!",
"The new phone I got is working really well.",
"I’m thinking of joining a fitness class next week.",
"I’ll need to review the budget before the meeting.",
"The kids are going to the park this afternoon.",
"I’m considering buying a new car this year.",
"Have you seen the new episode of that series?",
"I finished the report you asked for. It’s in your inbox.",
"Let me know if you need help with the presentation.",
"We need to decide on a venue for the party next month.",
"The coffee at this place is amazing!",
"I think I’ll take a day off this Friday.",
"I’ve been working on this project for a few weeks now.",
"I’m so happy we got tickets to that event!",
"I just got back from a long weekend trip to the mountains.",
"I’ve been spending some time reading and relaxing this weekend.",
"The dog looks happy today, he loves going for walks.",
"I’ll be at the office until 5 PM today.",
"We should get together soon to catch up!",
"I need to pick up a few items from the store later.",
"The weather has been perfect for gardening this week.",
    
"Let’s plan a barbecue for next Saturday!",
"I’ll send over the meeting notes as soon as I can.",
"Do you want to join me for a morning jog tomorrow?",
"I’ve got some work to finish before the weekend.",
"I’m so glad to hear you’re doing well!",
"Let’s grab a coffee tomorrow, it’s been a while!",
"I need to catch up on a few emails this evening.",
"The new coffee shop near my house is amazing.",
"I got a new book to read and I can’t wait to start!",
"I’ll be at the gym in about an hour.",
"I’m looking forward to the dinner party tonight.",
"I love the design of the new app, it's really easy to use.",
"We have a new project kickoff meeting on Monday.",
"I’ll take care of the grocery shopping for the week.",
"The kids have a soccer game this weekend, I’m so excited!",
"I’m thinking about switching to a new phone provider.",
"We should try the new sushi restaurant this weekend!",
"I have a Zoom call with the team in an hour.",
"We’re planning a picnic in the park tomorrow, join us!",
"My sister is coming over for a visit next weekend.",
"I need to update the website with the latest information.",
"Can you send me the details of that event you mentioned?",
"I’m thinking about going to a yoga class later.",
"I finally finished that book I’ve been reading for months!",
"Let’s meet at the new cafe that opened on Main Street.",
"I’ve just started watching that new show on Netflix.",
"I’ll be working from home tomorrow if you need anything.",
"I’ve been practicing my guitar skills lately.",
"We should book tickets for the play this weekend.",
"My parents are visiting me next month, so I’m looking forward to that.",
"I need to buy a new jacket for the winter season.",
"Let’s go hiking this Saturday, it’s the perfect weather.",
"I just joined a new fitness class and it’s been great.",
"I can’t wait for the concert tonight!",
"I’m working on a new project at work and it’s going well.",
"I’ll be at the library this afternoon for a study session.",
"I’m looking forward to my weekend getaway.",
"We’re going to the beach this Sunday, you should join us!",
"I’m meeting a friend for coffee later today.",
"Let’s make plans for next weekend; we haven’t caught up in a while.",
"I just signed up for a cooking class, I’m excited!",
"We’re thinking of adopting a pet soon.",
"I’m going for a walk in the park this afternoon.",
"I’m going to try that new fitness class tomorrow morning.",
"I’ve been really enjoying my morning routine lately.",
"Let me know if you need anything this afternoon.",
"I’m meeting with the team to discuss our new project proposal.",
"The event was amazing! I’m so glad I went.",
"I’m going to watch a movie tonight; I’ve been waiting for it to come out.",
"My brother is visiting next week, I’m excited to see him!",
"I’m organizing a game night this weekend, you should come!",
"I’ve been working on a new hobby—painting.",
"I’ll be heading to the airport early tomorrow morning.",
"I’m planning to spend the afternoon reading in the park.",
"I finally managed to get a reservation at that new restaurant.",
"I hope you're having a great day!",

"Let me know if you need any help with the project.",

"Can we catch up over coffee sometime this week?",

"I’ll meet you at the park in 20 minutes.",

"How was your weekend? We should hang out soon!",

"I just finished reading that book you recommended, it was amazing!",

"Can you send me the link for the meeting?",

"Do you want to join us for lunch tomorrow?",

"I’m working on a new project and would love your feedback.",

"Let’s plan a day trip to the beach this weekend.",

"I think I’ll stop by your place later, is that okay?",

"Have you seen the new movie that just came out?",

"I need to schedule a call to discuss the details.",

"I got a new book for my collection. Can’t wait to start reading!",

"We should go for a hike this weekend if the weather’s nice.",

"The dinner last night was so much fun, we should do it again!",

"I’ll be at the office at 9 AM tomorrow, looking forward to our meeting.",

"Can you please send me the updated version of the presentation?",

"How’s everything going with your new project?",

"I think we should try that new restaurant for lunch today!",

"The weather looks great for a walk, do you want to join?",

"I’ll send over the report by end of day today.",

"Let’s plan a movie night this Friday.",

"Can you help me with some edits on the document?",

"How did the meeting go this morning?",

"I’ll bring the snacks for the picnic tomorrow.",

"I’m looking forward to the concert this weekend!",

"I’ll need your feedback on the new design before we proceed.",

"Let me know when you’re free to chat about the next steps.",

"I’ve been thinking about signing up for a yoga class, want to join?",

"We should grab lunch next week, I miss catching up!",

"I’m looking forward to seeing you at the event tonight.",

"The new app looks great! I think it’s going to be really useful.",

"I need to pick up a few things from the store later, want to join?",

"I’ll be working from home tomorrow, so feel free to call me if needed.",

"Are you free this weekend? We should go for a walk in the park.",

"I’ve been really enjoying my morning routine lately.",

"It’s great to hear you’re doing well, we should chat soon.",

"Can you remind me to send that email this afternoon?",

"I’m planning to visit the museum on Saturday. You should come!",

"I’ll be attending the seminar tomorrow, I’ll update you afterward.",

"What time works best for a meeting this week?",

"I’m organizing a potluck dinner next weekend, you should come!",

"I’m thinking of starting a new fitness routine, want to join?",

"Do you have any recommendations for a good restaurant in the area?",

"Can you send me the agenda for tomorrow’s meeting?",

"We should definitely plan another trip together soon!",

"I’ve been reading a lot about personal finance lately, would love to share some tips!",

"I’ll update you on the project status later this afternoon.",

"I’m excited for the weekend, I’ve got a few fun plans lined up!",

"Let’s catch up soon, I miss our coffee dates.",

"I just joined a book club, it’s been so fun!",

"Do you want to come over for a casual dinner this weekend?",

"I’m thinking of starting a garden this spring, want to help?",

"It was great seeing you yesterday, let’s do it again soon!",

"I’ll be at the café at 3 PM, see you there!",

"What do you think about the new design for the website?",

"I just signed up for a photography class, super excited!",

"I’m planning to visit my family this weekend, it’ll be nice to relax.",

"I’ve been watching this amazing new series, you should check it out!",

"I’ll need your help reviewing this document before tomorrow’s meeting.",

"I’m so excited for the upcoming holiday trip, it’s going to be a blast!",

"I think we should go for a hike this weekend. What do you think?",

"I’m making dinner tonight, want to join?",

"I’ll call you later to discuss the details of the project.",

"I’ve just finished a great book! I think you’d love it.",

"I’m really enjoying my weekend so far, hope yours is great too!",

"Let’s plan a fun weekend getaway, it’s been way too long!",

"How’s your new job going? I’d love to hear all about it.",

"I’ll be working on the presentation tomorrow, can you send me your slides?",

"I’m planning to check out that new café later, want to join?",

"How about we meet up for brunch this Sunday?",

"I’m organizing a charity event, I’ll send you the details soon.",

"It’s so nice to have some free time this weekend. I’m planning to relax.",

"I’ve been meaning to tell you about this new hobby I picked up!",

"I’m planning to cook dinner tonight, care to join?",

"Let me know if you need help with anything this afternoon.",

"I’ve been working on a new project at work, I’m excited to share it with you.",

"I’ll send over the materials before the end of the day.",

"Do you want to go for a run in the morning?",

"I’ve been really enjoying my evening walks lately.",

"It was so great catching up with you yesterday, we should do it more often!",

"I’m making a list of weekend plans, any ideas?",

"I’ll be at the meeting at 10 AM sharp tomorrow.",

"It’s a beautiful day outside, I might go for a walk later.",

"I’ve been meaning to try that new workout class, want to come?",

"I’ll be home all afternoon if you want to swing by and chat.",

"I’m looking forward to our team call tomorrow.",

"I’m planning to join a new yoga class this week, it’s been on my list for a while!",

"Let’s make plans to go to the farmer’s market this weekend.",

"How’s everything going on your end? Let’s catch up soon.",

"I’ll meet you at the usual spot at 2 PM.",

"I’m planning to take a break from work this afternoon. How about you?",

"I’ve been thinking about picking up a new hobby, maybe painting.",

"How about we go for a bike ride this weekend?",

"I’ve been really into cooking lately, would love to try a new recipe with you!",

"I’ll update you on the project as soon as I get the latest information.",

"I’ve been spending some time reading, it’s been so relaxing!",

"We should try that new brunch place this weekend, heard it’s amazing!",

"I’m so excited for our meeting tomorrow, can’t wait to discuss the next steps.",
"Congratulations! You’ve won a free consultation worth $500, confirm your details to book!",

"We have an exclusive offer for you! Get a free trial membership to our fitness club, no strings attached!",

"Your subscription has been successfully activated. Please verify your address to receive your welcome gift!",

"You’ve earned a special discount! Just confirm your payment details to claim your reward!",

"Your order has been processed! Confirm your shipping address to receive it today.",

"Hurry! Claim your spot in our exclusive seminar. Limited seats available!",

"You're eligible for a complimentary gift, just fill out a quick form to claim it!",

"You’ve been selected for a special offer! Confirm your email for access to exclusive deals.",

"Limited time offer: You’ve won a free month of premium service. Just activate your trial now!",

"Congratulations! You’re one of the lucky winners of our giveaway, just confirm your identity to claim your prize!",

"We’re offering you a free gift! Simply provide your details to receive it!",

"You’ve been pre-approved for an exclusive membership, pay a small fee to complete your registration.",

"Your account has been successfully verified. Please confirm your contact details to receive your free rewards.",

"Exclusive offer: Sign up today and receive a complimentary service for one month!",

"You’ve been selected for a free product trial! Confirm your shipping address to get it delivered.",

"You’ve earned a reward for completing a survey! Confirm your bank details to claim it.",

"Your package is on the way! Confirm your shipping address to ensure prompt delivery.",

"Good news! You’ve received a reward for being a loyal customer. Claim your gift by confirming your details.",

"You’ve won an exclusive VIP experience! Just pay the small activation fee to secure your spot.",

"Hurry! Limited time offer to get a free upgrade. Confirm your details now!",

"Your subscription renewal is complete! Confirm your account details to enjoy uninterrupted service.",

"You’ve been selected for a free consultation. Confirm your availability to book your session!",

"Your product is eligible for a refund! Please provide your payment details to process the refund.",

"Congratulations! You've been selected for a premium membership, confirm your details to activate it!",

"You're one step away from your free gift! Confirm your phone number to finalize your order.",

"Your free trial has started! Simply confirm your billing information to continue your membership.",

"Congratulations! You’ve been pre-approved for a special deal, just confirm your account to unlock it.",

"You’ve won a limited-time offer! Confirm your registration to claim your exclusive access.",

"You’ve been selected for a special prize, confirm your details to claim it.",

"Your reward is waiting for you! Confirm your details to receive your exclusive bonus.",

"You’ve received a voucher for $200! Confirm your details to redeem it.",

"Congratulations! Your free trial of premium features is ready. Confirm your email to activate it.",

"You’ve been selected to receive a bonus! Confirm your bank details to claim your prize.",

"Exclusive offer: Receive a free service upgrade! Confirm your payment details to secure your spot.",

"Hurry! Confirm your details to claim your reward before time runs out!",

"You’ve earned an exclusive discount! Confirm your order now to receive the savings.",

"Your request has been approved! Confirm your details to finalize your order.",

"Special offer: Get a free service for one month! Confirm your details to activate it.",

"Your claim has been approved! Confirm your payment information to receive your refund.",

"You’ve won a prize! Confirm your details now to receive your gift!",

"You're eligible for a complimentary product! Confirm your address to have it shipped immediately.",

"Congratulations! You’ve unlocked a free gift. Confirm your shipping address to receive it.",

"You’ve won an exclusive reward! Confirm your payment details to receive your prize.",

"Your account has been verified! Confirm your address to receive your special gift.",

"Special offer: Claim your free consultation today, confirm your availability to secure your spot!",

"You’ve been selected to receive a surprise gift! Confirm your details to claim it.",

"Congratulations! You’ve unlocked an exclusive offer. Confirm your details to get started.",

... "You’ve earned a special reward for your loyalty! Confirm your email to receive it.",
... 
... "Your account has been upgraded! Confirm your details to activate your new features.",
... 
... "You’ve won a $500 voucher! Confirm your details to redeem it before it expires.",
... 
... ] 
... 
... labels = [1]*250 + [0]*250  
... 
... # ==== Train the model ====
... vectorizer = TfidfVectorizer()
... X = vectorizer.fit_transform(texts)
... y = labels
... 
... model = LogisticRegression()
... model.fit(X, y)
... 
... # ==== Streamlit UI ====
... st.set_page_config(page_title="Scam Detector AI", layout="centered")
... st.title("🕵️ Scam Message Detector")
... st.markdown("Enter a message below to detect whether it may be a **scam** or **not**.")
... 
... msg = st.text_area("📩 Message content", height=150)
... 
... if st.button("Analyze"):
...     if msg.strip():
...         vec = vectorizer.transform([msg])
...         pred = model.predict(vec)[0]
...         prob = model.predict_proba(vec)[0]
...         label = "🚨 **SCAM**" if pred == 1 else "✅ **Not Scam**"
...         st.markdown(f"### Result: {label}")
...         st.markdown(f"**Confidence:** {max(prob)*100:.2f}%")
...     else:
...         st.warning("Please enter a message to analyze.")
