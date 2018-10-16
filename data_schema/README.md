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

## random_ema.csv ##
0. status
1. smoke
2. when_smoke
3. when_eat
4. drink
5. when_drink
6. urge
7. cheerful
8. happy
9. angry
10. stress
11. sad
12. see_or_smell
13. access
14. smoking_location
15. participant_id
16. timestamp
17. offset
18. date
19. hour
20. minute
21. day_of_week

## eventcontingent-ema.csv ##
0. status
1. smoke
2. when_smoke
3. urge
4. cheerful
5. happy
6. angry
7. stress
8. sad
9. see_or_smell
10. access
11. smoking_location
12. participant_id
13. timestamp
14. offset
15. date
16. hour
17. minute
18. day_of_week

## eod-ema.csv ##
0. status
1. 8to9
2. 9to10
3. 10to11
4. 11to12
5. 12to13
6. 13to14
7. 14to15
8. 15to16
9. 16to17
10. 17to18
11. 18to19
12. 19to20
13. participant_id
14. timestamp
15. offset
16. date
17. hour
18. minute
19. day_of_week

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

