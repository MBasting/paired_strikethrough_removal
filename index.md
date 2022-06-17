# CycleGAN analysis for Image to Image Translation for Strikethrough Removal

Mark Basting-(M.J.Basting@student.tudelft.nl)
Sven de Witte-(S.dewitte@student.tudelft.nl)
Alexander Freeman-(A.J.Freeman@student.tudelft.nl)

Repository: https://github.com/MBasting/paired_strikethrough_removal

## Intro

Strikethrough removal is an interesting problem for both humans but also for deep learning algorithms. There are multiple ways to tackle the problem and in this blog we will reproduce and research the approaches from two papers from the last two years: *“Paired Image to Image Translation for Strikethrough Removal From Handwritten Words”* by Raphaela Heil, Ekta Vats and Anders Hast (Heil, 2022) and the prior  paper containing a CycleGAN approach from the same authors: *Strikethrough Removal from Handwritten Words Using CycleGANs*. (Heil, 2021). 

Both papers aim to solve the problem of strikethrough removal, either paired, where we have access to both linked clean and strikethrough ground-truth of single written words or unpaired, where we just have unrelated clean words and striked words.

![](https://i.imgur.com/wzkrMoj.png)

<p style="text-align: center;font-style: italic">Figure 1: various examples of strikethroughs</p>



What intrigued us about these two papers is the low score of the CycleGAN compared to the other standard algorithms, even though the parameter count of the model is much higher than the others. As per our understanding of the problem we thought that CycleGANs would be an adequate solution for this problem. 


## Prior research: the provided models

First, we start off by explaining the models that we are going to compare to reproduce the results. A good overview of the first four models can be seen in Figure 2[^1].

![](https://i.imgur.com/USIKDTC.png)

<p style="text-align: center;font-style: italic">Figure 2: The models from (Heil et al, 2022)</p> [1]

#### SimpleCNN

The first model is a relatively simple Convolutional Neural Network. It constists of three convolutional layers, then a densely connected network, followed by three upsample layers (or specifically ```nn.ConvTranspose2D``` layers). 

#### Shallow

Its design is based on the ResnetGenerator but no dedicated bottleneck layer is used. It consists of two convolutional down-, respectively upsampling layers. 

#### UNet

This architecture is a bit more complex and can best be explained by an example architecture image.  

![](https://i.imgur.com/gRJtVKc.png)

<p style="text-align: center;font-style: italic">Figure 3: The architecture of the UNet</p>

The contraction  is done by repeatedly applying convolutions, non-linear activation functions and pooling operations. During this operation we try to reduce the amount of features to more meaningful ones but in this process we lose spatial information. This is where the power of this network comes in. When we go up the U-shape, with upconvolution layers,  we try to combine both the feature information and the spatial information. In our case, the UNet design consists of one down and up-sampling as well as one bottleneck block which consists of four dense layers. 

#### Generator

This network is created and introduced in the other paper by the authors[^2]. It first contains three convolutional layers, followed by a densely connected network and finally three convolutional upsampling layers to form an end-to-end network. Each convolutional layer has a batch normalization step and ReLU. 

#### Attribute-guided CycleGAN

The CycleGAN is the most involved network of them all, which is also the reason we would expect it to function the best. A CycleGAN is an extension of a ‘traditional’ GAN (Generative Adversarial Network). And has been applied in other researches for similar tasks such as stain removal or watermark removal.

![](https://i.imgur.com/W3yjNcu.png)

<p style="text-align: center;font-style: italic">Figure 4: Architecture of CycleGAN</p>




### Parameter count

The amount of parameters can be found in table 1. In particular note the discrepancy between the CycleGAN and the other models. This at first is what drew our eye and therefore we decided to verify the results in a reproduction study.

| Model Name                | Parameter Count |
| ------------------------- | --------------: |
| SimpleCNN                 |          28 065 |
| Shallow                   |         154 241 |
| UNet                      |         181 585 |
| Generator                 |       1 345 217 |
| Attribute-Guided CycleGAN |       8 217 604 |

<p style="text-align: center;font-style: italic">Table 1: Parameter count obtained by authors of the reproduced paper.</p>


## Datasets


The paper itself used three datasets. First of all, there is Dracula_real[^6]. This is text that is written by a human for the book *Dracula* by Bram Stoker, with the handwritten word and the struck counterpart. The size is pretty limited with only 630 words in total.

Another dataset is the IAM_synth[^4] dataset. This is a *synthetic* dataset generated using a script that generated six types of strikethroughs: single line, double line, diagonal , cross, wave and zigzag. In total this dataset contains 4158 images.

Finally there is the Dracula_synth dataset[^7], which, like the name suggests, is also a synthetic dataset, but based on Dracula_real. The authors took the five training splits of Dracula_real of 126 clean images and for each sample generated one new struck sample, resulting in 5 * 126 images.

Practically, the IAM_synth dataset is mostly used to train on, while the Dracula_real set is used for validation.



## Reproduction

In both papers the displaying of the results is done quite differently. The plotting of the CycleGAN results were done using a boxplot over the different strikethroughs which additionally gave some insight into what type of strikethroughs are harder. Whereas in a later paper of the authors the results of the CycleGAN are compared to other Paired Image algorithms and the results are only shown in a table together with the standard deviation over 30 runs. 

The authors also left out the test setup used to do all these experiments and wrote their code in quite a complicated way to test it again. We rewrote a part of the code to be able to run more dynamically. 

After training the models on the original dataset based on IAM_synth and validating them on the Dracula_real dataset we got the results in table 2.


| Model     | F1(Theirs)              | F1(Ours)                | RMSE(Theirs)            | RMSE(Ours)              |
| --------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| SimpleCNN | 0.7204 (+/- 0.0303)     | 0.7503 (+/- 0.0065)     | **0.0827 (+/- 0.0038)** | 0.0875 (+/- 0.0005)     |
| Shallow   | 0.7540 (+/- 0.0028)     | 0.7409 (+/- 0.0063)     | 0.0932 (+/- 0.0044)     | 0.0915 (+/- 0.0004)     |
| UNet      | 0.7451 (+/- 0.0013)     | 0.7387 (+/- 0.0075)     | 0.1005 (+/- 0.0033)     | 0.1108 (+/- 0.0007)     |
| Generator | **0.7577 (+/- 0.0035)** | **0.7552 (+/- 0.0069)** | 0.0921 (+/- 0.0021)     | 0.0921 (+/- 0.0006)     |
| CycleGAN  | 0.7189 (+/- 0.0243)     | 0.7176 (+/- 0.0061)     | 0.0927 (+/- 0.0212)     | **0.0517 (+/- 0.0002)** |

<p style="text-align: center;font-style: italic">Table 2: Our results of training on IAM_synth and validating on Dracula_real. Mean and standard deviation are taken over 30 runs. Best in bold.</p>

Reproducing the other table by training and testing on splits of IAM_synth provides the results in table 3.

| Model     | F1(Theirs)              | F1(Ours)                | RMSE(Theirs)            | RMSE(Ours)              |
| --------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| SimpleCNN | 0.8727 (+/- 0.0042)     | **0.9602 (+/- 0.0013)** | 0.0753 (+/- 0.0025)     | 0.0389 (+/- 0.0002)     |
| Shallow   | 0.9163 (+/- 0.0045)     | 0.8982 (+/- 0.0050)     | 0.0558 (+/- 0.0025)     | 0.0663 (+/- 0.0006)     |
| UNet      | 0.9599 (+/- 0.0015)     | 0.9354 (+/- 0.0049)     | 0.0301 (+/- 0.0012)     | 0.0474 (+/- 0.0010)     |
| Generator | **0.9697 (+/- 0.0012)** | 0.9544 (+/- 0.0028)     | **0.0237 (+/- 0.0016)** | **0.0379 (+/- 0.0005)** |
| CycleGAN  | 0.7981 (+/- 0.0284)     | 0.8159 (+/- 0.0091)     | 0.1172 (+/- 0.0286)     | 0.0697 (+/- 0.0013)     |

<p style="text-align: center;font-style: italic">Table 3: Our results of training and testing on splits of IAM_synth. Mean and standard deviation are taken over 30 runs. Best in bold.</p>


What is interesting in these results especially is that the CycleGAN has way more variance present in its results and actually outperforms other methods when the RMSE is used but lacks behind with the F1 score. 

To further compare these results we can look at the resulting images generated by the different models from [^1].

![](https://i.imgur.com/zqSS5Ba.png)

<p style="text-align: center;font-style: italic">Figure 5: outputs for each of the model and each of the strikethrough types. </p>

As mentioned, in one of their works they showed a boxplot over the different strikethroughs with the F1 score for every strikethrough together with their respective mean and standard deviation, they however left out this analysis when comparing different networks in their later paper. We performed the same analysis and repeated the process for the CycleGAN model. In our repository you can find the generated figures for all dataset combinations. 

As seen below the results showed the same type of distribution between the different algorithms, for the different stroke types. When comparing the different stroke types it can be seen that the more ocluded and less readable words do peform worse. 


<img style="align: left;" src="https://i.imgur.com/imI879m.png" alt="graph" width="60%"/>

<p style="text-align: center;font-style: italic">Figure 6: RMSE score for models trained and evaluated on the IAMsynth dataset. Lower is better.</p>

<img style="align: right;" src="https://i.imgur.com/JRqu22T.png" alt="graph" width="60%"/>

<p style="text-align: center;font-style: italic">Figure 7: F1 score for models trained and evaluated on the IAMsynth dataset. Higher is better. </p>


<img style="align: center;" src="https://i.imgur.com/HPVXZrg.png" alt="graph" width="60%"/>

<p style="text-align: center;font-style: italic">Figure 8: F1 score of trained and evaluated CycleGANs with different setups on the IAMsynth dataset as showcased in the paper. Higher is better.</p>

Comparing our figure with their figure for the F1 scores we can see that there are quite some difference. The error margin of our results are significantly bigger than the original results and overall the resulting mean for all strikethrough types is also almost always lower. 


## Hypothesis

After analyzing the previous work, the code and our own reproductions of the result using the original code, we formalized two different hypotheses / improvements to improve the CycleGAN algorithm.

For the first improvement we expected that the size of the dataset that the original paper was using was too small for the CycleGAN to effectively learn the problem. Our hypothesis was “The CycleGAN will perform better than the other algorithms when trained on a large dataset”. To test this we created a bigger paired dataset that is more than a 100 times bigger than the original dataset being used. We then ran all the algorithms with this dataset and compared the results.

After analyzing the performance of the generator we saw that it was unable to correctly generate accurate strikethrough images. We noticed that the balance between the generator and discriminator was off. Meaning our discriminator out performed the generator by a large margin. Our hypothesis for this was “The generator takes the mean of the different stroke types, resulting in the generator not being able to replicate an accurate strikethrough image”. To test this hypothesis we ran the CycleGAN using a single strikethrough type. We then compared the results of the generator and the overall algorithm with that of the original algorithm.

To further investigate the performance influence of certain parameters a hyper parameter experiment was performed. In this experiment the batchSize, weight of the identity loss and weight of the clean generator were varied. This last hyperparameter was chosen because earlier analysis of the intermediate results of the CycleGAN showed that it could not generate sensible strikethrough images. An example of such a result can be seen below. The repository contains a tool to visualize the intermediate validation results by generating a video over the epochs.  

![](https://i.imgur.com/JxO12ML.png)

![](https://i.imgur.com/ab0mKSg.png)

<p style="text-align: center;font-style: italic">Figure 9: Outputs of the internal clean2struck generator. As can be seen, no actual strikethroughs are applied here.</p>


After analyzing the code and researching different state of the art CycleGANs and their application we hypothesized that the loss functions being used for the CycleGAN were not fit to solve this problem. We tested this by tuning the hyperparameters for the current loss functions and tried to see if we can change the loss functions entirely. [^1]

### The loss functions

To analyze the performance the performance of the CycleGAN more in depth we decided to perform a small hyperparameter tuning experiment. We did this by varying the batch size between 2 and 4, the identity lambda between 0, 0.5 and 1 and the clean lambda between 10, 15 and 20. The clean lambda is especially interesting in this case as we noticed earlier that the strikethrough generator could not generate sensible images representing images close to real strikethrough words. 

The difference in performance between the different runs with different combinations of hyperparameters where minimal. Seen below are two graphs from one of the runs. As can be seen the loss of the discriminator goes down quite fast which indicates that the discriminator still easily outperforms the generator. The graphs below are representable for almost all runs done in the ablation study. 

<img src="https://i.imgur.com/wQFv9Hs.png" style="width:48%;" />
<img src="https://i.imgur.com/txxSnES.png" style="width:48%;" />

<p style="text-align: center;font-style: italic">Figure 10: The graphs illustrating the problem with the generator.</p>

We wanted to change the loss functions from the current L2 loss to a loss function that maybe optimizes better for the removal of pixels instead of the greyscale it currently optimized towards. This ended up to not be very fruitful and therefore we will not go into detail here.


### More data equals better?

To test the hypothesis "*The CycleGAN will perform better than the other algorithms when trained on a large dataset*" we had to create a bigger dataset ourselves.

To expand, we first looked at using the IAMonDo-database[^8]. This, however, used an InkML format which made it very hard to work with. At the same time, while we were analyzing the data used in the original paper, we noticed that the IAM_synth database was only generated from a small part of the original IAM database. We used the full IAMSynth word database to create 404.830 strikethrough images [^5]. 

To create the paired strike-through images we used an altered version of the strikethrough generation tool used by the authors in the paper to keep consistency[^3]. For every word in the original IAMSynth database 6 strikethrough images were created, all with a different kind of strikethrough. This data was then divided in to 70% training data,  10% validation data and 20% test data. 

Since this dataset is huge it was difficult to train the models for the full 30 epochs and due to time-constraints we decided to train each model for 8 epochs. Intermediate validation was done each 4 epochs, compared to every 2 epochs in other analysis. 


| Model                     | RMSE      | F1        |
| ------------------------- | --------- | --------- |
| SimpleCNN                 | **0.022** | 0.978     |
| Shallow                   | 0.045     | 0.942     |
| Unet                      | 0.031     | 0.964     |
| Generator                 | **0.022** | **0.980** |
| Attribute-Guided CycleGAN | 0.056     | 0.791     |

<p style="text-align: center;font-style: italic">Table 4: Our results of training and testing on our IAM_synth full dataset. </p>

Table 4 shows very similar results to our reproduction results. Again the generator outperforms other models with SimpleCNN getting close second. Our hypothesis that the CycleGAN would perform better with more data is hereby disproven. 

### Fixing the discriminator with more aligned data

Next up was the hypothesis about the misalignment of the clean2struck generator on the multiple types of strikethrough. We tested the hypothesis: "The generator takes the mean of the different stroke types, resulting in the generator not being able to replicate an accurate strikethrough image".

We tested this hypothesis by running the CycleGAN on only the diagonal Strikethrough type. This way the generator could not take a mean of different strikethrough types if it was doing this. 

This resulted in an RMSE score of 0.049051, and F1 score of 0.866676. 

![](https://i.imgur.com/U18cPCr.png)

<p style="text-align: center;font-style: italic">Figure 11: Handpicked example of generated strikethrough sample. </p>

![](https://i.imgur.com/zlC2wMu.png)

<p style="text-align: center;font-style: italic">Figure 12: Handpicked examples of cleaned sample.</p>

Figure 11 and Figure 12 show that the strikethrough generator still has difficulties in generating meaningful strikethrough samples. However, the cleaned images show quite promising results. 

## Img2Img: a Visual Transformer

After analyzing the CycleGAN, we wondered if some kind of Visual Transformer existed and if so, if it would perform on this problem as well. As it turns out, a paper containing exactly that was published in October of 2021 [^9]. Their code is also available on GitHub.

Since the code for the transformer is wired to use squared images and our images aren't all square, we had to to some data processing. We finally decided to use a 512x512 image as input. This was to keep it a power of two and to keep the parameter count low. We removed images bigger than either dimension from IAM_synth and upscaled all other images using a constant white color. We also had to modify the convolutional part of the network to actually use these new sizes, since the default is 256x256 and the network is not actually parameterized on these sizes themselves.

The vision transformer was trained for 30 epochs (similarly to the original reproduction results) on the IAM_synth dataset. Below you can see a cherrypicked output and a lemonpicked output for each Strikethrough type. As you can see in the images, some bad examples contain block artifacts. 

Ideally, we would have wanted to evaluate this vision Transformer further, since it shows very promising results. However, we did not have the time for this analysis. 

#### Single line 

<img src="https://i.imgur.com/wbbwvnZ.png" style="width:60%;" />
<img src="https://i.imgur.com/duMnNjU.png" style="width:60%;" />

#### Double line

<img src="https://i.imgur.com/pdt8Kd1.png" style="width:60%;" />
<img src="https://i.imgur.com/fzXmJ9l.png" style="width: 60%;" />

#### Diagonal

<img src="https://i.imgur.com/pL0A9Z0.png" style="width:60%;" />
<img src="https://i.imgur.com/Mtqfkj4.png" style="width:60%;" />

#### Cross

<img src="https://i.imgur.com/dNWC1Tu.png" style="width:60%;" />
<img src="https://i.imgur.com/vI7F7N4.png" style="width:60%;" />

#### Wave

<img src="https://i.imgur.com/dUW2Lpq.png" style="width:60%;" />
<img src="https://i.imgur.com/k4Hrx34.png" style="width:60%;" />

#### ZigZag

<img src="https://i.imgur.com/ryhrhtb.png" style="width:60%;" />
<img src="https://i.imgur.com/7vLb5yE.png" style="width:60%;" />

#### Scratch

<img src="https://i.imgur.com/ERBKtVa.png" style="width:60%;" />
<img src="https://i.imgur.com/MJxQLJc.png" style="width:60%;" />


## Discussion and future work

All results indicate that the CycleGAN just doesn't perform as well as the other models on this task. Our analysis indicated that a potential issue could be that the discriminator outperforms the generator and thus more performance could be gained when these are correctly paired. We also found that the strikethrough generator does not give meaningful output, we expected that solving this problem, by only focussing on one strikehthrough would result in a better overal performance of the clean generator. However, only focussing on one strikethrough type did not solve this issue. It did however result in a better RMSE and F1 score to our surprise. 

It also needs to be said that the CycleGAN has a more difficult task in that it receives unpaired images compared to the other models (apart from the Vision Transformer). This makes the task of removing strikethrough types a lot harder. Furthermore, CycleGAN was originally used to perform style-transfer problems, the problem it is trying to tackle here is quite a bit different. 

The Vision Transformer showed promising results when trained on a relatively small dataset (IAM_synth), we did not however have the time to evaluate the F1 score or RMSE score on this dataset and thus have only performed a visual analysis. We leave this as future work. Another direction that could be explored is running the Vision Transformer on our own fully generated IAM_synth dataset, note that this can take quite a significant amount of time and computation. 

A point to note is that we don't know if the Transformer underfits or (more likely) overfits, since we did not perform validation for the transformer. This is especially important since the scratch strikethroughs are removed too, while the other models have significant problems with those problems. This could indicate that the model is overfitting on the problem set and remembering original inputs for the images that actually function well and produces bad output for the ones it does not.


[^1]: Heil, R., Vats, E., & Hast, A. (2022). Paired Image to Image Translation for Strikethrough Removal From Handwritten Words. doi:10.48550/ARXIV.2201.09633
[^2]: Heil, R., Vats, E., & Hast, A. (2021). Strikethrough Removal from Handwritten Words Using CycleGANs. Στο J. Lladós, D. Lopresti, & S. Uchida (Επιμ.), 16th International Conference on Document Analysis and Recognition, ICDAR 2021, Lausanne, Switzerland, September 5-10, 2021, Proceedings, Part IV (σσ. 572–586). doi:10.1007/978-3-030-86337-1_38
[^3]: Heil, Raphaela. (2021). RaphaelaHeil/strikethrough-generation: Release for publication (v1.0). Zenodo. https://doi.org/10.5281/zenodo.4767063
[^4]: Heil, Raphaela, Vats, Ekta, & Hast, Anders. (2021). IAM Strikethrough Database (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4767095
[^5]: U. Marti and H. Bunke. 2002. The IAM-database: An English Sentence Database for Off-line Handwriting Recognition. Int. Journal on Document Analysis and Recognition, Volume 5, pages 39 - 46, 2002.https://fki.tic.heia-fr.ch/databases/iam-handwriting-database#icdar02
[^6]: Heil, Raphaela, Vats, Ekta, & Hast, Anders. (2021). Single-Writer Strikethrough Dataset (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4765063
[^7]: Heil, Raphaela, Vats, Ekta, & Hast, Anders. (2022). Single-Writer Synthetic Strikethrough Dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6406539
[^8]: Indermühle, Emanuel.(2011).IAM Online Document Database (IAMonDo-database)http://www.iapr-tc11.org/mediawiki/index.php?title=IAM_Online_Document_Database_(IAMonDo-database)
[^9]: Gündüç, Y.. (2021). Tensor-to-Image: Image-to-Image Translation with Vision Transformers. 
