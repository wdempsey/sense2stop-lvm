# Sense2Stop: Smoking detection via hierarchical latent marked point processes  #

This file describes how I(Jason) think the data should be organized.

In essence, we should organize the data by the following hierarchical structure:

1. User.
2. Type of data.
3. Relevant columns from relevant raw data files extracted and combined together.

Following this paradigm, here is how I think we should organize the data:

## puff_probability.csv ##
Schema:
0. UserId
1. puff_probability
2. raw_timestamp
3. "day of week"
4. "month"
5. "hour"

We should think about if we want to put every user into one csv or separate each user into his/her own file.

## puff_episode.csv ##
Schema:
0. UserId
1. puff_event
2. raw_timestamp
3. "day of week"
4. "month"
5. "hour"

Again, We should think about if we want to put every user into one csv or separate each user into his/her own file.

## EMAs ##
Each user should have 3 files that record the data from the 3 different EMAs separately. One way to do this could be just "userid_random_EMA.csv", "userid_smoking_EMA.csv", and "userid_EOD_EMA.csv".

Then, in each of the file, I would organize the data by the following schema:
0. userid
1. raw_timestamp (this could either be the "startime" or "end_timestamp" field in the original dataset)
2. status
3. EMA_id
4. [An array of questions, where each question is a dictionary]

Question:
{"question_text": "",
"response": "",
"question_id": "",
"finish time": "" (-1 in this field indicates no answer)
}

We should convert raw responses to interpretable numerical reponses. That is, some responses are categorical indicators and we need to make them numbers.

