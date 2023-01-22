"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import csv
from PIL import Image
import re
import contractions # for word contractions
import string
import emoji

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/TFIDF.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	tab1, tab2, tab3, tab4= st.tabs(["Information", "Prediction", "About Titans", "Feedback"])
	sdf = pd.DataFrame(columns = ['ref', 'Email address', 'Ratings', 'Suggestions'])
	
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "About Titans"]
	#selection = st.sidebar.selectbox("Choose Option", options)
	
	# Building out the "Information" page
	#if selection == "Information":
	with tab1:
		logo = Image.open("resources/imgs/logo1.png")
		st.image(logo, use_column_width=True)
	# Creates a main title and subheader on your page -
	# these are static across all pages
		st.title("CUDAR")
		st.subheader("Learn about climate change")

		st.info("### General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("The sky was once a brilliant shade of blue, with fluffy white clouds drifting lazily across its expanse. The air was clean and fresh, and the world was alive with the sounds of chirping birds and buzzing insects.")
		img2 = Image.open("resources/imgs/green lush.jpg")
		img2 = img2.resize((300,200))
		st.image(img2)
		st.write("But then, something began to change. The sky grew dark and hazy, and the clouds turned a sickly shade of grey. The air grew thick and heavy, and the once-lively world grew quiet and still.")
		st.write("Climate change had arrived, and with it came a multitude of woes. The earth grew warmer, and the oceans rose to swallow up low-lying lands. Storms grew stronger and more frequent, devastating communities and wiping out entire ecosystems.")
		st.write("Animals struggled to adapt to the changing environment, and many species went extinct. Those that survived faced new challenges, as their habitats were destroyed and their food sources dwindled.")
		img1 = Image.open("resources/imgs/climate pic 1.jpg")
		st.image(img1)
		st.write("The humans, too, felt the effects of climate change. Crops failed, and the earth grew barren and inhospitable. Water became scarce, and conflicts erupted over its control.")
		img3 = Image.open("resources/imgs/climate pic 2.jpg")
		st.image(img3)
		st.write("But it wasn't all doom and gloom. Some people recognized the dire situation and took action. They worked tirelessly to reduce their carbon footprint and to find ways to reverse the damage that had been done.")
		st.write("They planted trees, installed solar panels, and developed new technologies to capture and store carbon. And slowly but surely, the world began to heal.")
		st.write("The sky cleared, and the air grew fresh once more. The animals returned, and the earth was alive with the sounds of life once again. And the humans, grateful for their second chance, vowed to do better and to protect the earth for generations to come.")
		
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	#if selection == "Prediction":
	with tab2:
		#st.markdown("scroll down")
		logo = Image.open("resources/imgs/logo1.png")
		st.image(logo, use_column_width=True)
	# Creates a main title and subheader on your page -
	# these are static across all pages
		st.title("CUDAR")
		st.subheader("Know your climate change sentiment")
		pri_color = st.get_option("theme.primaryColor")
		txt_color = st.get_option("theme.textColor")
		bg_color = st.get_option("theme.backgroundColor")
		sec_color = st.get_option("theme.secondaryBackgroundColor")
		font = st.get_option("theme.font")

		pri_color = "#c7288e"
		txt_color = "#fdfdfd"
		bg_color = "#085192"
		sec_color = "#abc1ec"
		font = "serif"
		st.info("### Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("What do you want to tweet?","Type Here")

		if st.button("Classify with SVM Model"):
			#options1 = ["Logistic Regression model", "Other models"]
			#selection1 = st.button.selectbox("Choose Option", options1)
			#if selection == "Logistic Regression model":
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# vect_text = tweet_text
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/svm_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			#st.success("Text Categorized as: {}".format(prediction))
			if prediction == 2:
				st.success('This is a News article tweet with links to factual news about climate change') 
			elif prediction == 1:
				st.success('This tweet is Pro Climate Change : supporting the belief of man-made climate change')
				st.balloons()
			elif prediction == 0:
				st.success('This tweet is Neutral : neither supports nor refutes the belief of man-made climate change')
			elif prediction == -1:
				st.success(' This tweet is Anti : does not believe in man-made climate change')
		
		if st.button('Classify with Logistic Regression model'):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# vect_text = tweet_text
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/logistics_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if prediction == 2:
				st.success('This is a News article tweet with links to factual news about climate change') 
			elif prediction == 1:
				st.success('This tweet is Pro Climate Change: supporting the belief of man-made climate change')
				st.balloons()
			elif prediction == 0:
				st.success('This tweet is Neutral: neither supports nor refutes the belief of man-made climate change')
			elif prediction == -1:
				st.success(' This tweet is Anti: does not believe in man-made climate change')

		if st.button("Classify with DecisionTree Model"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# vect_text = tweet_text
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/dtc_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			#st.success("Text Categorized as: {}".format(prediction))
			if prediction == 2:
				st.success('This is a News article tweet with links to factual news about climate change') 
			elif prediction == 1:
				st.success('This tweet is Pro Climate Change : supporting the belief of man-made climate change')
				st.balloons()
			elif prediction == 0:
				st.success('This tweet is Neutral : neither supports nor refutes the belief of man-made climate change')
			elif prediction == -1:
				st.success(' This tweet is Anti : does not believe in man-made climate change')
		




	#if selection == "About Titans":
	with tab3:
		logo = Image.open("resources/imgs/logo1.png")
		st.image(logo, use_column_width=True)
	# Creates a main title and subheader on your page -
	# these are static across all pages
		st.title("CUDAR")
		#st.subheader("Know your climate change sentiment")
		st.info("### Meet the team")
		# You can read a markdown file from supporting resources folder
		st.markdown("### Data Scientist - Chima Enyeribe")
		jasper = Image.open("resources/imgs/jasper.jpg")
		jasper = jasper.resize((300,300))
		st.image(jasper)
		st.markdown("A contemplative solutionist with years of relevant industry experience.Created a variety of models to optimize processes, saving time and resources for Titans Inc. He lives in Kaduna, Nigeria with his wife and kids")
		st.markdown("MSc. Computer Science, University of the People, California (USA)")
		st.markdown("### Machine Learning Engineer - Umar Kabir")
		umar = Image.open("resources/imgs/umar.jpg")
		umar = umar.resize((300,300))
		st.image(umar)
		st.markdown("He has implemented isolated statistical analysis/ML/AI model into high-performance, high-availability production level systems that provide quick and easy access to all interested users or stakeholders. Happily married with 2 kids")
		st.markdown("MSc. Statistics, Ahmadu Bello University, Zaria (Nigeria)")
		st.markdown("### Data Analyst - Akor Christian")
		akor_chris = Image.open("resources/imgs/akor_chris.jpg")
		akor_chris = akor_chris.resize((300,300))
		st.image(akor_chris)
		st.markdown("He's passionate about diving into complex data and producing strategic recommendations. Have worked on a number of projects dealing with database management systems, emerging technology and business trends. He has designed and built over 30 statistical analysis models and increased our data collection and processing rates by 120%. He lives in PortHarcourt with his wife.")
		st.markdown("Ph.D. Mathematics, University of Nigeria, Nsukka (Nigeria)")
		#st.markdown("### Business Analyst - Christian Thompson")
		#ct = Image.open("resources/imgs/climate pic 1.jpg")
		#st.image(ct)
		#st.markdown("Experienced Business Analyst with 8+ years experience boosting warehouse profitability by 50 percent for a  Fortune 100 e-commerce retailer. Seeking to use people management and analytic skills at Titans Corporate. He is single and searching.")
		#st.markdown("Ph.D. Computer Science, MBA, University of Something (Finland)")
		st.markdown("### Data Engineer - Daniel Uwaoma")
		dan = Image.open("resources/imgs/dan.jpg")
		dan = dan.resize((300,300))
		st.image(dan)
		
		st.markdown("Maintained data pipeline up-time of 99.8 percent while ingesting streaming and transactional data across 8 different primary data sources using Spark, RedShift, S3 and Python. He's getting married soon ! Congrats!")
		st.markdown("BSc. Mechanical Engineering, MSc. Robotics, University of Lagos (Nigeria)")


	with tab4:
		logo = Image.open("resources/imgs/logo1.png")
		st.image(logo, use_column_width=True)
	# Creates a main title and subheader on your page -
	# these are static across all pages
		st.title("CUDAR")
		st.subheader("How can we serve you better?")
		st.info("### Tell us what you think of our app.")
		with open('ts.csv', 'w') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['Email address', 'Ratings', 'Suggestions'])
		sug = pd.read_csv("suggest1.csv")
		suggestions_df = pd.DataFrame(columns = ['Email address', 'Ratings', 'Suggestions'])
		#st.write(sug)

		fb=st.form("fb", clear_on_submit = True)
		email = fb.text_input("Your Email")
		rate = fb.slider('Rate us on a scale of 1 to 10', 1, 10, 5)
		st.write("You rated us a ", rate)
		list_of_suggestions = []
			#add_ref = 1
		add_ref = sug['Ref'].max()+1
		suggestions = fb.text_area("Type your suggestions here")
		submit = fb.form_submit_button('Submit your suggestions')
			#suggestions_df['Email']= email
			#suggestions_df['ratings']= rate
			#suggestions_df['suggestions'] = suggestions
			#suggestions_df = pd.DataFrame()
		if submit:
			new_data = {'Ref': add_ref, 'Email address':email, 'Ratings':int(rate), 'Suggestions': suggestions}
			nd = pd.DataFrame({'ref': [add_ref], 'Email address':[email], 'Ratings':[rate], 'Suggestions': [suggestions]})
			sug.append(new_data, ignore_index=True)
			
		#st.write(sug)
		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
