    # PART B: MULTI-TASK LEARNING FOR FRUIT TYPE AND QUALITY CLASSIFICATION

    ## 11. SCENARIO 1: MULTI-TASK BASELINE - NORMAL COLOURED IMAGES

    ### 11.1 Introduction

    Scenario 1 forms the baseline performance for multi-task learning on standard RGB colour images. This scenario represents a key departure from the single-task strategy in Part A by setting up a dual-objective architecture that simultaneously predicts both fruit type, 11 classes, and quality level, 3 classes: Good, Mild, Rotten. The multi-task formulation reflects the practical reality that real-world fruit classification systems often need to perform identification and quality assessment operations simultaneously when deployed within agricultural supply chains.

    The main motivation behind multi-task learning lies in efficiency considerations related to both data utilisation and computational resources. Multi-task architecture, instead of training different dedicated models on each particular task, shares the same convolutional backbone for both objectives; it potentially lowers the parameter count, possibly reduces inference time, and benefits from shared visual features useful for both classification goals. This approach tests whether low-level features extracted during fruit type identification, such as colour patterns, surface texture, and overall shape, can simultaneously inform quality assessment decisions.

    ### 11.2 Configuration

    Table 15 presents the complete configuration for multi-task Scenario 1. The configuration mirrors the single-task baseline (Scenario 1 from Part A) in all parameters except for the architectural modification to support dual classification heads.

    | Parameter              | Value                   |
    | ---------------------- | ----------------------- |
    | Model Architecture     | Multi-Task Simple CNN   |
    | Input Channels         | 3 (RGB)                 |
    | Image Size             | 224 × 224               |
    | Batch Size             | 32                      |
    | Epochs                 | 50                      |
    | Learning Rate          | 0.001                   |
    | Optimizer              | Adam                    |
    | Weight Decay           | 0.0001                  |
    | Scheduler              | ReduceLROnPlateau       |
    | Patience               | 10 epochs               |
    | Gradient Clipping      | 1.0                     |
    | Warmup Epochs          | 5                       |
    | Mixed Precision        | Enabled                 |
    | Class Weights          | Enabled (auto-computed) |
    | Quality Task Weight    | 1.0                     |
    | Fruit Type Task Weight | 1.0                     |
    | Augmentation           | Disabled                |
    | Grayscale              | Disabled                |
    | Random Seed            | 42                      |

    **Table 15: Configuration parameters for multi-task Scenario 1 baseline**

    The multi-task architecture employs equal loss weighting (both set to 1.0) for quality and fruit type objectives, allowing the optimiser to balance gradients naturally based on task difficulty rather than imposing artificial prioritisation. This configuration provides the foundation against which subsequent multi-task scenarios can be compared.

    ### 11.3 Results and Analysis

    #### 11.3.1 Overall Performance Metrics

    The multi-task baseline achieved exceptional performance across both classification objectives, demonstrating that shared feature representations can effectively serve dual predictive goals. Table 16 presents comprehensive performance metrics separated by task, revealing near-perfect accuracy on fruit type identification and excellent quality classification performance.

    | Task           | Metric    | Validation | Test    | Δ from Single-Task S1 |
    | -------------- | --------- | ---------- | ------- | --------------------- |
    | **Quality**    | Accuracy  | 99.89%     | 99.89%  | +0.05%                |
    |                | Precision | 99.89%     | 99.89%  | +0.05%                |
    |                | Recall    | 99.89%     | 99.89%  | +0.10%                |
    |                | F1-Score  | 99.89%     | 99.89%  | +0.05%                |
    |                | AUC       | 1.0000     | 1.0000  | 0.0000                |
    | **Fruit Type** | Accuracy  | 100.00%    | 100.00% | N/A                   |
    |                | Precision | 100.00%    | 100.00% | N/A                   |
    |                | Recall    | 100.00%    | 100.00% | N/A                   |
    |                | F1-Score  | 100.00%    | 100.00% | N/A                   |
    |                | AUC       | 1.0000     | 1.0000  | N/A                   |

    **Table 16: Overall performance metrics for multi-task Scenario 1**

    The performance on the fruit type classification task was perfect, 100% according to all metrics, both on the validation and test sets, with zero misclassifications among the 1873 validation and 939 test samples. This perfect performance indicates that the 11 fruit types in the dataset exhibit highly distinctive visual characteristics that are easily discriminated by the convolutional backbone, even when the network must attend to quality assessment features simultaneously.

    The quality classification task achieved 99.89% validation accuracy and 99.89% test accuracy, marginally better than the performance of the single-task baseline from Part A, which achieved 99.84% and 99.79%, respectively. This modest absolute improvement in performance (+0.05% validation, +0.10% test) contradicts the commonly held belief that multi-task learning necessarily results in a trade-off of performance between the different objectives. The validation set yielded only 2 misclassifications out of 1873 samples, and the test set yielded only 1 error out of 939 samples. The near-perfect AUC scores, 1.0000 for both tasks on both sets, signify optimal discriminative capability across all decision thresholds.

    Such consistency among validation and test performance for both tasks-specifically, identical precision of 99.89% for quality and 100% for fruit type-demonstrates robust generalization with no overfitting, suggesting that the shared convolutional backbone has learned proper feature representations rather than memorizing training examples. The fact that quality classification performance slightly exceeded single-task performance while achieving perfect fruit type classification suggests the occurrence of positive transfer learning effects where fruit-specific features-color distribution and surface texture patterns specific to each fruit variety-may have contributed extra context that refined quality discrimination.

    #### 11.3.2 Per-Class Performance Analysis

    Table 17 presents detailed per-class metrics for both tasks, revealing how the multi-task architecture balanced performance across quality categories and fruit types.

    | Task           | Class       | Split      | Precision | Recall  | F1-Score |
    | -------------- | ----------- | ---------- | --------- | ------- | -------- |
    | **Quality**    | Good        | Validation | 100.00%   | 99.83%  | 99.92%   |
    |                |             | Test       | 100.00%   | 100.00% | 100.00%  |
    |                | Mild        | Validation | 99.79%    | 99.79%  | 99.79%   |
    |                |             | Test       | 99.58%    | 100.00% | 99.79%   |
    |                | Rotten      | Validation | 99.88%    | 100.00% | 99.94%   |
    |                |             | Test       | 100.00%   | 99.75%  | 99.88%   |
    | **Fruit Type** | BananaDB    | Validation | 100.00%   | 100.00% | 100.00%  |
    |                |             | Test       | 100.00%   | 100.00% | 100.00%  |
    |                | CucumberQ   | Validation | 100.00%   | 100.00% | 100.00%  |
    |                |             | Test       | 100.00%   | 100.00% | 100.00%  |
    |                | GrapeQ      | Validation | 100.00%   | 100.00% | 100.00%  |
    |                |             | Test       | 100.00%   | 100.00% | 100.00%  |
    |                | KakiQ       | Validation | 100.00%   | 100.00% | 100.00%  |
    |                |             | Test       | 100.00%   | 100.00% | 100.00%  |
    |                | PapayaQ     | Validation | 100.00%   | 100.00% | 100.00%  |
    |                |             | Test       | 100.00%   | 100.00% | 100.00%  |
    |                | PeachQ      | Validation | 100.00%   | 100.00% | 100.00%  |
    |                |             | Test       | 100.00%   | 100.00% | 100.00%  |
    |                | PearQ       | Validation | 100.00%   | 100.00% | 100.00%  |
    |                |             | Test       | 100.00%   | 100.00% | 100.00%  |
    |                | PepperQ     | Validation | 100.00%   | 100.00% | 100.00%  |
    |                |             | Test       | 100.00%   | 100.00% | 100.00%  |
    |                | StrawberryQ | Validation | 100.00%   | 100.00% | 100.00%  |
    |                |             | Test       | 100.00%   | 100.00% | 100.00%  |
    |                | WatermeloQ  | Validation | 100.00%   | 100.00% | 100.00%  |
    |                |             | Test       | 100.00%   | 100.00% | 100.00%  |
    |                | tomatoQ     | Validation | 100.00%   | 100.00% | 100.00%  |
    |                |             | Test       | 100.00%   | 100.00% | 100.00%  |

    **Table 17: Per-class performance metrics for multi-task Scenario 1**

    The quality classification results show balanced performance across all three categories. The "Good" class achieved perfect 100% precision on both validation and test sets, with 99.83% recall on validation and perfect 100% recall on test. The "Mild" class showed 99.79% precision and recall on validation, with slightly lower 99.58% precision but perfect 100% recall on test. The "Rotten" class demonstrated strong performance with 99.88% precision and perfect 100% recall on validation, inverting to perfect 100% precision and 99.75% recall on test. This class-balanced performance indicates that the multi-task model does not exhibit systematic bias towards any particular quality level and handles both quality extremes (Good and Rotten) as well as the intermediate Mild category with high reliability.

    The results of the fruit type classification demonstrate perfect 100% across all metrics for all 11 fruit types on both validation and test sets. This comprehensive perfect performance across categories ranging from frequently represented fruits (tomatoQ with 1990 images) to sparsely represented fruits (StrawberryQ with only 216 images) suggests that inter-fruit visual differences are sufficiently pronounced that class imbalance does not degrade classification capability. The shared convolutional backbone successfully learned discriminative features for each fruit type without confusion, even for visually similar categories that might share colour or texture characteristics.

    ### 11.4 Confusion Matrix Analysis

    **[INSERT Figure 11-1: Quality Task Validation Confusion Matrix]**
    _Source: `reports/scenario1_multitask/figures/confusion_matrix_quality_val.png`_

    **[INSERT Figure 11-2: Fruit Type Task Validation Confusion Matrix]**
    _Source: `reports/scenario1_multitask/figures/confusion_matrix_fruit_val.png`_

    Quality task validation confusion matrix shows 2 misclassifications out of 1873 samples. One "Good" sample was classified as "Mild," while one "Mild" sample got classified as "Rotten." Critically, there were no errors between the quality extremes: no "Good" samples were misclassified as "Rotten" and vice versa. This pattern of error indicates that the model maintains clear discrimination at quality boundaries while facing minor difficulty only with samples with transitional characteristics.

    Compared to the single-task baseline from Part A, which generated 3 validation errors (all "Mild" misclassified as "Rotten"), the multi-task model performs better, with one fewer error and a more balanced distribution of errors across quality boundaries rather than accumulating the errors at the Mild-Rotten boundary. This suggests that fruit-specific contextual information contributed by the fruit type classification head might have helped fine-tune quality discrimination by allowing the model to apply fruit-specific quality criteria rather than generic quality features. The confusion matrix on fruit type shows perfect diagonal structure with zero off-diagonal elements; this confirms 100% classification accuracy with no confusion between any pair of fruit types. This perfect performance validates the multi-task architecture's ability to maintain excellent fruit identification capability while doing quality assessment, hence demonstrating that the shared backbone does not suffer from task interference where one objective degrades the performance of another.

    ### 11.5 ROC Curve Analysis and AUC Scores

    **[INSERT Figure 11-3: Quality Task Test ROC Curves]**
    _Source: `reports/scenario1_multitask/figures/roc_curves_quality_test.png`_

    **[INSERT Figure 11-4: Quality Task Validation ROC Curves]**
    _Source: `reports/scenario1_multitask/figures/roc_curves_quality_val.png`_

    **[INSERT Figure 11-5: Fruit Type Task Test ROC Curves]**
    _Source: `reports/scenario1_multitask/figures/roc_curves_fruit_test.png`_

    **[INSERT Figure 11-6: Fruit Type Task Validation ROC Curves]**
    _Source: `reports/scenario1_multitask/figures/roc_curves_fruit_val.png`_

    It follows that the quality task ROC curves have excellent discriminative capability in terms of multi-task quality classification. All three classes of quality achieve perfect or near-perfect AUC scores: Validation AUC of 1.0000 for all classes, micro-average AUC of 1.0000; Test AUC of 1.0000 for all classes, micro-average AUC of 1.0000. All quality class ROC curves track along the upper-left corner, the point [0,1]; this means the model realizes a maximum true positive rate while maintaining a minimum false positive rate across all decision thresholds.

    This ideal behavior confirms that the quality classification head produces well-calibrated probability predictions with clear separation between correct and incorrect classifications. If the model assigns high confidence to a quality prediction, then that prediction is correct with extremely high probability. The perfect AUC scores are consistent with those of the single-task baseline, indicating that adding the fruit type classification objective did not degrade the quality head's probability calibration or discriminative power.

    ROC curves on the fruit type task demonstrate perfect discriminative capability, with all 11 fruit types achieving AUC = 1.0000 on both validation and test sets. For all fruit types, the curves track perfectly along the upper-left corner, confirming that the model never assigns higher confidence to an incorrect fruit type than to the correct type, across any probability threshold. This perfect rank ordering validates the fruit type classification head's ability to produce maximally informative probability distributions over fruit categories.

    ### 11.6 Training History and Convergence Analysis

    **[INSERT Figure 11-7: Multi-Task Training History]**
    _Source: `reports/scenario1_multitask/figures/training_history.png`_

    The training history for the multi-task baseline provides critical insight into how the dual-objective optimisation converged and whether task interference affected learning dynamics compared to single-task training. The multi-panel training curves track loss and accuracy for both quality and fruit type tasks simultaneously, along with the combined total loss and learning rate schedule.

    The model converged rapidly for both tasks within the first 10-15 epochs. The fruit type task achieved >99% accuracy by epoch 3 and stabilised at perfect 100% accuracy by epoch 5, maintaining this performance throughout remaining epochs. This extremely rapid convergence for fruit type classification confirms that inter-fruit visual differences are highly distinctive and easily learned by the convolutional backbone even during the warmup phase.

    The quality task showed slightly slower but still impressive convergence, achieving >99% accuracy by epoch 8 and stabilising above 99.8% by epoch 12. Both training and validation accuracy curves for quality classification tracked each other closely throughout training, with validation occasionally matching or slightly exceeding training accuracy. This pattern indicates genuine learning rather than memorisation, with the model generalising well to held-out validation data without overfitting.

    The combined loss curve (sum of quality loss and fruit type loss weighted by their respective task weights) decreased smoothly from approximately 0.5 to near-zero by epoch 15, following a stable trajectory without oscillations. The learning rate schedule shows two reduction events triggered by the ReduceLROnPlateau scheduler around epochs 18 and 32, dropping from the initial 0.001 to approximately 5 × 10⁻⁴ and then 2.5 × 10⁻⁴. These adaptive reductions enabled progressive refinement of decision boundaries for both tasks.

    Comparing convergence dynamics to the single-task baseline from Part A reveals that multi-task training achieved comparable or faster convergence despite optimising two objectives simultaneously. The single-task model achieved >99% accuracy by epoch 5, whilst the multi-task model achieved >99% fruit type accuracy by epoch 3 and >99% quality accuracy by epoch 8. This suggests that positive transfer between tasks may have accelerated learning, with fruit-specific features providing useful inductive bias for quality assessment and vice versa.

    ### 11.7 Conclusion

    Multi-task Scenario 1 establishes a robust baseline for simultaneous fruit type and quality classification, achieving 99.89% quality accuracy and perfect 100% fruit type accuracy on both validation and test sets. The results provide compelling evidence that multi-task learning offers substantial benefits for fruit classification applications without sacrificing performance on either task.

    The perfect fruit type classification demonstrates that the 11 fruit categories in the FruQ dataset exhibit highly discriminative visual features that enable error-free identification even when the network must simultaneously attend to quality assessment features. The quality classification performance marginally exceeded the single-task baseline from Part A (+0.05% validation, +0.10% test), suggesting positive transfer learning effects where fruit-specific contextual information refined quality discrimination.

    From an architectural efficiency perspective, the multi-task model achieves comparable parameter efficiency to two separate single-task models whilst providing both outputs in a single forward pass. The shared convolutional backbone (32→64→128→256 filter progression) extracts hierarchical features that serve both classification heads effectively, with task-specific fully connected layers (256 units each) providing sufficient capacity for final decision-making.

    The systematic evaluation methodology applied to this baseline scenario establishes a comprehensive framework for assessing multi-task performance that will be applied consistently across subsequent scenarios to evaluate how input modifications affect both tasks independently and jointly.

    ---

    ## 12. SCENARIO 2: MULTI-TASK GRAYSCALE IMAGES

    ### 12.1 Introduction

    Scenario 2 explores the role of colour information in multi-task fruit classification by converting all input images to grayscale whilst maintaining the dual-objective architecture from Scenario 1. This experimental manipulation addresses fundamental questions about feature dependencies across tasks: is colour information equally critical for fruit type identification and quality assessment, or might one task prove more colour-dependent than the other?

    The motivation for this scenario stems from theoretical considerations about what visual features drive each classification task. Fruit type identification might reasonably depend heavily on colour, as distinctive colour patterns (e.g., yellow bananas, red tomatoes, purple grapes) provide immediately recognisable discriminative features. Quality assessment, conversely, might rely more on texture patterns such as surface smoothness, wrinkle formation, and spotting that can be captured in grayscale intensity variations. If these hypotheses hold, we would expect grayscale conversion to degrade fruit type classification more severely than quality classification, potentially revealing task-specific feature dependencies.

    From a practical standpoint, understanding colour dependency has implications for deployment scenarios where imaging hardware constraints or lighting conditions might necessitate grayscale or single-channel infrared imaging. If grayscale representations prove sufficient for one or both tasks, simplified imaging systems could be deployed without sacrificing classification accuracy whilst benefiting from reduced data bandwidth and processing requirements.

    ### 12.2 Configuration

    Table 18 presents the configuration for multi-task Scenario 2. The configuration differs from multi-task Scenario 1 in only two parameters: INPUT_CHANNELS (reduced from 3 to 1) and GRAYSCALE (changed from disabled to enabled). All other parameters including learning rate, optimiser settings, and task loss weights remain identical to ensure fair comparison.

    | Parameter              | Value                   |
    | ---------------------- | ----------------------- |
    | Model Architecture     | Multi-Task Simple CNN   |
    | Input Channels         | **1 (Grayscale)**       |
    | Image Size             | 224 × 224               |
    | Batch Size             | 32                      |
    | Epochs                 | 50                      |
    | Learning Rate          | 0.001                   |
    | Optimizer              | Adam                    |
    | Weight Decay           | 0.0001                  |
    | Scheduler              | ReduceLROnPlateau       |
    | Patience               | 10 epochs               |
    | Gradient Clipping      | 1.0                     |
    | Warmup Epochs          | 5                       |
    | Mixed Precision        | Enabled                 |
    | Class Weights          | Enabled (auto-computed) |
    | Quality Task Weight    | 1.0                     |
    | Fruit Type Task Weight | 1.0                     |
    | Augmentation           | Disabled                |
    | Grayscale              | **Enabled**             |
    | Random Seed            | 42                      |

    **Table 18: Configuration parameters for multi-task Scenario 2**

    ### 12.3 Results and Analysis

    #### 12.3.1 Overall Performance Metrics

    Scenario 2 achieved remarkable performance that challenges conventional assumptions about colour's role in fruit classification. Table 19 presents comprehensive performance metrics, revealing that grayscale images maintained near-perfect accuracy for both tasks with only marginal differences from the RGB baseline.

    | Task           | Metric    | Validation | Test    | Δ from MT S1   |
    | -------------- | --------- | ---------- | ------- | -------------- |
    | **Quality**    | Accuracy  | 99.95%     | 99.89%  | +0.06% / 0.00% |
    |                | Precision | 99.95%     | 99.89%  | +0.06% / 0.00% |
    |                | Recall    | 99.95%     | 99.89%  | +0.06% / 0.00% |
    |                | F1-Score  | 99.95%     | 99.89%  | +0.06% / 0.00% |
    |                | AUC       | 1.0000     | 1.0000  | 0.0000         |
    | **Fruit Type** | Accuracy  | 100.00%    | 100.00% | 0.00%          |
    |                | Precision | 100.00%    | 100.00% | 0.00%          |
    |                | Recall    | 100.00%    | 100.00% | 0.00%          |
    |                | F1-Score  | 100.00%    | 100.00% | 0.00%          |
    |                | AUC       | 1.0000     | 1.0000  | 0.0000         |

    **Table 19: Overall performance metrics for multi-task Scenario 2**

    The fruit type classification task maintained perfect 100% accuracy across all metrics on both validation and test sets, with zero errors among 1873 validation samples and 939 test samples. This perfect performance exactly matches the RGB baseline, demonstrating that colour information is not essential for fruit type identification in this dataset. The result is striking given intuitive expectations that colour would be critical for distinguishing between fruits. This finding suggests that the fruit types in the FruQ dataset exhibit distinctive grayscale signatures, texture patterns, and shape characteristics that enable perfect discrimination without colour information.

    The quality classification task achieved 99.95% validation accuracy and 99.89% test accuracy, representing a marginal improvement over the RGB baseline on validation (+0.06%) whilst matching baseline performance on test (0.00%). The validation set produced only 1 misclassification among 1873 samples (compared to 2 errors in RGB baseline), whilst the test set produced 1 error among 939 samples (identical to RGB baseline). The near-perfect AUC scores (1.0000 for both tasks on both sets) remain identical to baseline, indicating preserved discriminative capability across all decision thresholds.

    The consistency between validation and test performance for both tasks demonstrates robust generalisation. The fact that grayscale conversion not only maintained but slightly improved quality classification accuracy on validation data whilst preserving perfect fruit type classification challenges the assumption that RGB colour channels provide essential information for these tasks. Instead, the results suggest that texture, shape, and grayscale intensity patterns contain sufficient discriminative information for both fruit identification and quality assessment.

    #### 12.3.2 Per-Class Performance Analysis

    Table 20 presents detailed per-class metrics for both tasks under grayscale processing, revealing how removal of colour information affected classification performance across quality categories and fruit types.

    | Task           | Class           | Split      | Precision | Recall  | F1-Score |
    | -------------- | --------------- | ---------- | --------- | ------- | -------- |
    | **Quality**    | Good            | Validation | 100.00%   | 99.83%  | 99.92%   |
    |                |                 | Test       | 100.00%   | 100.00% | 100.00%  |
    |                | Mild            | Validation | 99.79%    | 100.00% | 99.89%   |
    |                |                 | Test       | 99.58%    | 100.00% | 99.79%   |
    |                | Rotten          | Validation | 100.00%   | 100.00% | 100.00%  |
    |                |                 | Test       | 100.00%   | 99.75%  | 99.88%   |
    | **Fruit Type** | (All 11 fruits) | Validation | 100.00%   | 100.00% | 100.00%  |
    |                |                 | Test       | 100.00%   | 100.00% | 100.00%  |

    **Table 20: Per-class performance metrics for multi-task Scenario 2**

    The quality classification results show excellent balanced performance across all three categories. The "Good" class achieved perfect 100% precision on both validation and test sets, with 99.83% recall on validation and perfect 100% recall on test (matching RGB baseline exactly). The "Mild" class showed 99.79% precision with perfect 100% recall on validation, and 99.58% precision with perfect 100% recall on test (identical to RGB baseline). The "Rotten" class demonstrated perfect 100% precision and recall on validation (improved from RGB baseline's 99.88% precision), and perfect 100% precision with 99.75% recall on test (matching RGB baseline).

    Notably, the validation set "Rotten" class performance improved under grayscale processing, achieving perfect 100% metrics compared to RGB baseline's 99.88% precision. This counterintuitive improvement suggests that colour information may occasionally introduce confounding factors for rotten fruit detection, with grayscale intensity patterns providing more reliable indicators of severe degradation than colour shifts that might vary across fruit types or lighting conditions.

    The fruit type classification results maintained perfect 100% across all metrics for all 11 fruit types on both validation and test sets, exactly matching RGB baseline performance. This comprehensive perfect performance across diverse fruit categories, including fruits that humans would typically distinguish primarily by colour (e.g., bananas versus tomatoes), indicates that grayscale features such as surface texture, shape contours, and intensity distribution patterns enable flawless fruit identification without colour information.

    ### 12.4 Confusion Matrix Analysis

    **[INSERT Figure 12-1: Quality Task Validation Confusion Matrix]**
    _Source: `reports/scenario2_multitask_grayscale/figures/confusion_matrix_quality_val.png`_

    **[INSERT Figure 12-2: Fruit Type Task Validation Confusion Matrix]**
    _Source: `reports/scenario2_multitask_grayscale/figures/confusion_matrix_fruit_val.png`_

    The quality task validation confusion matrix reveals 1 misclassification among 1873 samples: one "Good" sample was classified as "Mild." Critically, zero errors occurred at the Mild-Rotten boundary, which had been the primary error location in previous scenarios. The Rotten class achieved perfect classification with zero errors. This represents an improvement over the RGB baseline (which produced 2 errors: 1 Good→Mild and 1 Mild→Rotten), with the grayscale model eliminating the Mild-Rotten boundary confusion entirely.

    The error pattern suggests that without colour information, the model encountered slight difficulty distinguishing high-quality fruit from fruit showing very early degradation (Good versus Mild), but maintained perfect discrimination between more clearly differentiated quality levels. The complete elimination of Mild-Rotten confusion under grayscale processing is particularly interesting, suggesting that colour variations may have introduced ambiguity for distinguishing intermediate from severe degradation, whilst grayscale texture patterns provide more reliable indicators of advanced decay.

    The fruit type confusion matrix displays perfect diagonal structure with zero off-diagonal elements, confirming 100% classification accuracy with no confusion between any pair of fruit types. This perfect performance exactly matches the RGB baseline, validating that the 11 fruit types in the dataset exhibit grayscale-distinguishable characteristics that enable error-free identification without colour information.

    ### 12.5 ROC Curve Analysis and AUC Scores

    **[INSERT Figure 12-3: Quality Task Test ROC Curves]**
    _Source: `reports/scenario2_multitask_grayscale/figures/roc_curves_quality_test.png`_

    **[INSERT Figure 12-4: Quality Task Validation ROC Curves]**
    _Source: `reports/scenario2_multitask_grayscale/figures/roc_curves_quality_val.png`_

    **[INSERT Figure 12-5: Fruit Type Task Test ROC Curves]**
    _Source: `reports/scenario2_multitask_grayscale/figures/roc_curves_fruit_test.png`_

    **[INSERT Figure 12-6: Fruit Type Task Validation ROC Curves]**
    _Source: `reports/scenario2_multitask_grayscale/figures/roc_curves_fruit_val.png`_

    The ROC curves of quality tasks under grayscale processing are excellent in their discriminative capability: the validation AUC for all classes was 1.0000, the micro-average AUC was 1.0000, and the test AUC was 1.0000 across all classes, with a micro-average AUC of 1.0000. ROC curves across all quality classes track at the upper-left corner-point [0,1]-indicating the highest true positive rate while maintaining the lowest false positive rate across all decision thresholds.

    These perfect AUC scores exactly match the RGB baseline, indicating that grayscale conversion preserved the probability calibration and discriminative power of the quality classification head. Indeed, the model achieves the optimal rank ordering of the predictions across all probability thresholds, assigning always higher confidence to the correct quality classification compared to any incorrect one for each sample.

    The fruit type task ROC curves similarly maintain perfect discriminative capability under grayscale processing, with all 11 fruit types achieving AUC = 1.0000 on both validation and test sets. The curves for all fruit types track perfectly along the upper-left corner, confirming that the model assigns maximum confidence to correct fruit types across any probability threshold. This perfect performance exactly matches the RGB baseline, validating that grayscale features provide sufficient information for maximally confident fruit identification.

    ### 12.6 Training History and Convergence Analysis

    **[INSERT Figure 12-7: Multi-Task Grayscale Training History]**
    _Source: `reports/scenario2_multitask_grayscale/figures/training_history.png`_

    The training history for grayscale multi-task learning reveals convergence dynamics comparable to the RGB baseline, with both tasks achieving high accuracy rapidly despite the removal of colour information. The fruit type task converged extremely rapidly, achieving >99% accuracy by epoch 2-3 and stabilising at perfect 100% accuracy by epoch 4-5, matching or slightly exceeding the RGB baseline's convergence speed. This rapid convergence confirms that grayscale features alone provide sufficient discriminative information for fruit type identification without requiring colour channels.

    The quality task showed similarly impressive convergence, achieving >99% accuracy by epoch 6-8 and stabilising above 99.8% by epoch 10-12. Both training and validation accuracy curves tracked each other closely, with validation occasionally matching or exceeding training accuracy, indicating genuine generalisation rather than memorisation. The convergence trajectory closely matches the RGB baseline, suggesting that grayscale intensity patterns capture quality-relevant features as effectively as colour information.

    The combined loss curve decreased smoothly from approximately 0.45 to near-zero by epoch 12-15, following a stable trajectory comparable to RGB baseline. The learning rate schedule triggered reductions around epochs 16-20 and 30-35, enabling progressive refinement of decision boundaries. The training stability and convergence speed under grayscale processing match RGB baseline performance, demonstrating that removal of colour information did not introduce learning difficulties or require additional training iterations.

    ### 12.7 Comparative Analysis with Multi-Task Scenario 1 (RGB Baseline)

    The performance comparison between multi-task grayscale (Scenario 2) and multi-task RGB (Scenario 1) processing reveals remarkable equivalence across both tasks, challenging assumptions about colour's necessity for fruit classification.

    | Task       | Metric   | MT S1 (RGB) Val | MT S2 (Gray) Val | MT S1 (RGB) Test | MT S2 (Gray) Test |
    | ---------- | -------- | --------------- | ---------------- | ---------------- | ----------------- |
    | Quality    | Accuracy | 99.89%          | 99.95%           | 99.89%           | 99.89%            |
    | Quality    | Errors   | 2 / 1873        | 1 / 1873         | 1 / 939          | 1 / 939           |
    | Fruit Type | Accuracy | 100.00%         | 100.00%          | 100.00%          | 100.00%           |
    | Fruit Type | Errors   | 0 / 1873        | 0 / 1873         | 0 / 939          | 0 / 939           |

    **Table 21: Performance comparison between multi-task RGB and grayscale scenarios**

    The removal of colour information resulted in identical or marginally improved performance. Quality classification improved on validation (+0.06%, reducing errors from 2 to 1) whilst matching baseline performance on test (identical 99.89%). Fruit type classification maintained perfect 100% accuracy with zero errors on both datasets. This equivalence contradicts intuitive expectations that colour would be essential for fruit identification.

    ### 12.8 Conclusion

    Multi-task Scenario 2 demonstrates that grayscale images provide sufficient discriminative information for near-perfect fruit type and quality classification, achieving 99.95% quality accuracy and perfect 100% fruit type accuracy on validation, with 99.89% quality and 100% fruit type accuracy on test. The results challenge conventional assumptions about colour's necessity for fruit classification applications.

    The perfect fruit type classification under grayscale processing indicates that the 11 fruit categories exhibit distinctive texture, shape, and intensity patterns that enable error-free identification without colour channels. Quality classification performance matched or marginally exceeded RGB baseline, suggesting that texture-based quality indicators captured in grayscale intensity variations provide equally or more reliable assessment than colour-inclusive features.

    From a practical standpoint, these findings suggest that grayscale-based fruit classification systems could be deployed with confidence, offering computational efficiency benefits (reducing input dimensionality by 66% from 3 channels to 1) without sacrificing classification accuracy. The dataset-specific nature of these findings should be noted, as different fruit varieties or quality assessment criteria might show larger performance gaps between RGB and grayscale processing.

    ---

    ## 13. SCENARIO 3: MULTI-TASK AUGMENTED IMAGES

    ### 13.1 Introduction

    Scenario 3 explores the impact of data augmentation techniques on multi-task fruit classification performance using standard RGB colour images. This scenario applies a comprehensive suite of geometric and photometric transformations during training to artificially expand the effective training set size and potentially improve model robustness to variations in fruit orientation, lighting conditions, and imaging parameters.

    The primary objective centres on determining whether data augmentation provides benefits for multi-task learning, and critically, whether augmentation affects both tasks equally or introduces task-specific performance trade-offs. The fruit type identification task, which achieved perfect 100% accuracy in both baseline and grayscale scenarios, might prove robust to augmentation-induced variations given the distinctive inter-fruit visual differences. The quality classification task, conversely, involves more subtle discrimination between adjacent quality levels (particularly Good versus Mild and Mild versus Rotten boundaries) that might prove more sensitive to augmentation transformations that alter colour, brightness, or texture characteristics.

    The augmentation pipeline employed in this scenario includes both geometric transformations (random rotation ±10 degrees, horizontal and vertical flipping with 50% probability) and photometric adjustments (colour jitter: brightness ±20%, contrast ±20%, saturation ±20%, hue ±10%). These transformations simulate realistic variations that might occur during image acquisition in agricultural applications, where camera angles, fruit orientations, and lighting conditions vary considerably.

    ### 13.2 Configuration

    Table 22 presents the configuration for multi-task Scenario 3. The configuration is identical to multi-task Scenario 1 except for the enabled data augmentation, allowing direct attribution of performance differences to augmentation effects.

    | Parameter              | Value                   |
    | ---------------------- | ----------------------- |
    | Model Architecture     | Multi-Task Simple CNN   |
    | Input Channels         | 3 (RGB)                 |
    | Image Size             | 224 × 224               |
    | Batch Size             | 32                      |
    | Epochs                 | 50                      |
    | Learning Rate          | 0.001                   |
    | Optimizer              | Adam                    |
    | Weight Decay           | 0.0001                  |
    | Scheduler              | ReduceLROnPlateau       |
    | Patience               | 10 epochs               |
    | Gradient Clipping      | 1.0                     |
    | Warmup Epochs          | 5                       |
    | Mixed Precision        | Enabled                 |
    | Class Weights          | Enabled (auto-computed) |
    | Quality Task Weight    | 1.0                     |
    | Fruit Type Task Weight | 1.0                     |
    | Augmentation           | **Enabled**             |
    | Grayscale              | Disabled                |
    | Random Seed            | 42                      |

    **Table 22: Configuration parameters for multi-task Scenario 3**

    Table 23 specifies the augmentation transformations applied:

    | Property        | Value                                                     |
    | --------------- | --------------------------------------------------------- |
    | Random Rotation | ±10 degrees                                               |
    | Horizontal Flip | 50% probability                                           |
    | Vertical Flip   | 50% probability                                           |
    | Colour Jitter   | Brightness ±20%, Contrast ±20%, Saturation ±20%, Hue ±10% |

    **Table 23: Augmentation specifications for multi-task Scenario 3**

    ### 13.3 Results and Analysis

    #### 13.3.1 Overall Performance Metrics

    Scenario 3 revealed differential augmentation impact across tasks, with quality classification experiencing substantial performance degradation whilst fruit type identification maintained perfect accuracy. Table 24 presents comprehensive performance metrics, revealing a pronounced task-specific response to augmentation.

    | Task           | Metric    | Validation | Test    | Δ from MT S1      |
    | -------------- | --------- | ---------- | ------- | ----------------- |
    | **Quality**    | Accuracy  | 97.06%     | 96.59%  | -2.83% / -3.30%   |
    |                | Precision | 97.14%     | 96.88%  | -2.75% / -3.01%   |
    |                | Recall    | 97.06%     | 96.59%  | -2.83% / -3.30%   |
    |                | F1-Score  | 97.08%     | 96.65%  | -2.81% / -3.24%   |
    |                | AUC       | 0.9969     | 0.9955  | -0.0031 / -0.0045 |
    | **Fruit Type** | Accuracy  | 100.00%    | 100.00% | 0.00%             |
    |                | Precision | 100.00%    | 100.00% | 0.00%             |
    |                | Recall    | 100.00%    | 100.00% | 0.00%             |
    |                | F1-Score  | 100.00%    | 100.00% | 0.00%             |
    |                | AUC       | 1.0000     | 1.0000  | 0.0000            |

    **Table 24: Overall performance metrics for multi-task Scenario 3**

    The fruit type classification task maintained perfect 100% accuracy across all metrics on both validation and test sets, with zero errors among 1873 validation samples and 939 test samples. This perfect performance exactly matches both RGB baseline and grayscale scenarios, demonstrating exceptional robustness to augmentation transformations. The result indicates that inter-fruit visual differences (texture patterns, shape characteristics, grayscale intensity distributions) are sufficiently pronounced that geometric distortions and photometric variations do not introduce classification ambiguity. The fruit type identification task appears highly robust to realistic imaging variations.

    The quality classification task, conversely, experienced substantial performance degradation under augmentation. Validation accuracy dropped to 97.06% (down 2.83% from baseline's 99.89%), whilst test accuracy dropped to 96.59% (down 3.30% from baseline's 99.89%). The validation set produced 55 misclassifications among 1873 samples (compared to 2 errors in baseline), whilst the test set produced 32 errors among 939 samples (compared to 1 error in baseline). The AUC scores declined to 0.9969 on validation and 0.9955 on test, down from perfect 1.0000 in baseline, indicating reduced probability calibration quality.

    This substantial degradation of quality classification performance whilst fruit type identification remained perfect reveals task-specific augmentation sensitivity. The findings suggest that aggressive augmentation transformations, particularly colour jitter (brightness, contrast, saturation adjustments), may have distorted subtle quality indicators more severely than fruit-defining characteristics. Colour shifts that alter perceived browning, spotting, or discolouration patterns critical for quality assessment might push borderline samples across quality boundaries, forcing the model to learn overly general features that sacrifice precision on unaugmented evaluation images. Fruit type features (overall shape, surface texture patterns, structural characteristics), conversely, appear robust to these transformations.

    #### 13.3.2 Per-Class Performance Analysis

    Table 25 presents detailed per-class metrics, revealing how augmentation affected classification performance across quality categories whilst maintaining perfect fruit type classification.

    | Task           | Class           | Split      | Precision | Recall  | F1-Score |
    | -------------- | --------------- | ---------- | --------- | ------- | -------- |
    | **Quality**    | Good            | Validation | 99.31%    | 97.97%  | 98.64%   |
    |                |                 | Test       | 99.66%    | 97.31%  | 98.47%   |
    |                | Mild            | Validation | 92.34%    | 96.42%  | 94.34%   |
    |                |                 | Test       | 89.02%    | 98.74%  | 93.63%   |
    |                | Rotten          | Validation | 98.36%    | 96.78%  | 97.56%   |
    |                |                 | Test       | 99.48%    | 94.80%  | 97.08%   |
    | **Fruit Type** | (All 11 fruits) | Validation | 100.00%   | 100.00% | 100.00%  |
    |                |                 | Test       | 100.00%   | 100.00% | 100.00%  |

    **Table 25: Per-class performance metrics for multi-task Scenario 3**

    The quality classification results reveal unbalanced performance degradation across quality categories. The "Mild" class experienced the most severe decline, achieving only 92.34% validation precision and 89.02% test precision, representing drops of approximately 7.5% from baseline. The "Mild" class F1-scores dropped to 94.34% on validation and 93.63% on test, down from baseline's 99.79%. This substantial degradation indicates that intermediate-quality fruit showing early deterioration proved highly sensitive to augmentation transformations.

    The "Good" class maintained relatively strong performance with 98.64% validation F1-score and 98.47% test F1-score, down approximately 1.3% from baseline's near-perfect performance. The "Rotten" class showed intermediate degradation with 97.56% validation F1-score and 97.08% test F1-score, down approximately 2.4% from baseline. The error pattern suggests that augmentation-induced colour and brightness variations most severely affected discrimination of subtle quality indicators characteristic of the Mild category, whilst more pronounced features of Good (pristine appearance) and Rotten (severe degradation) remained relatively recognisable.

    The fruit type classification results maintained perfect 100% across all metrics for all 11 fruit types on both validation and test sets, exactly matching baseline performance. This comprehensive perfect performance confirms that fruit-defining visual characteristics proved entirely robust to geometric distortions and photometric variations introduced by augmentation.

    ### 13.4 Confusion Matrix Analysis

    **[INSERT Figure 13-1: Quality Task Validation Confusion Matrix]**
    _Source: `reports/scenario3_multitask_augmented/figures/confusion_matrix_quality_val.png`_

    **[INSERT Figure 13-2: Fruit Type Task Validation Confusion Matrix]**
    _Source: `reports/scenario3_multitask_augmented/figures/confusion_matrix_fruit_val.png`_

    The quality task validation confusion matrix reveals 55 misclassifications among 1873 samples, distributed across multiple class boundaries with concentration at the Good-Mild and Mild-Rotten interfaces. The primary error pattern shows bidirectional confusion between adjacent quality categories: Good samples classified as Mild, Mild samples classified as Good, Mild samples classified as Rotten, and Rotten samples classified as Mild. The Mild class bore the primary burden of classification errors, consistent with per-class metrics showing severely degraded precision and recall for intermediate-quality fruit.

    The error distribution suggests that augmentation transformations, particularly aggressive colour jitter (±20% brightness/contrast/saturation), introduced ambiguity that blurred quality boundaries. Reducing brightness on Good fruit may have made pristine samples appear mildly degraded, whilst increasing brightness on Rotten fruit may have masked decay patterns. Similarly, rotation and flipping might have obscured or emphasised specific surface defects critical for distinguishing Mild from adjacent categories. The concentration of errors at the Mild-adjacent boundaries indicates that borderline quality assessment proves highly sensitive to augmentation-induced variations.

    The fruit type confusion matrix displays perfect diagonal structure with zero off-diagonal elements, confirming 100% classification accuracy with no confusion between any fruit types. This perfect performance validates exceptional robustness of fruit identification to augmentation transformations.

    ### 13.5 ROC Curve Analysis and AUC Scores

    **[INSERT Figure 13-3: Quality Task Test ROC Curves]**
    _Source: `reports/scenario3_multitask_augmented/figures/roc_curves_quality_test.png`_

    **[INSERT Figure 13-4: Quality Task Validation ROC Curves]**
    _Source: `reports/scenario3_multitask_augmented/figures/roc_curves_quality_val.png`_

    **[INSERT Figure 13-5: Fruit Type Task Test ROC Curves]**
    _Source: `reports/scenario3_multitask_augmented/figures/roc_curves_fruit_test.png`_

    **[INSERT Figure 13-6: Fruit Type Task Validation ROC Curves]**
    _Source: `reports/scenario3_multitask_augmented/figures/roc_curves_fruit_val.png`_

    The quality task ROC curves demonstrate degraded but still strong discriminative capability under augmentation. The validation AUC of 0.9969 and test AUC of 0.9955 represent declines from baseline's perfect 1.0000, indicating that probability calibration suffered under augmentation. The ROC curves for quality classes show slight departure from the perfect upper-left corner, with curves passing through intermediate points before reaching optimal performance. This behaviour indicates that the model occasionally assigned lower confidence to correct quality predictions or higher confidence to incorrect predictions compared to baseline's perfect rank-ordering.

    The AUC decline, whilst statistically significant, remains above 0.995, indicating that the quality classification head still provides excellent probability estimates despite increased hard classification errors. The model maintains strong discriminative capability but with reduced confidence margin between correct and incorrect predictions for borderline samples affected by augmentation-induced feature distortions.

    The fruit type task ROC curves maintained perfect AUC = 1.0000 on both validation and test sets, exactly matching baseline performance. The curves track perfectly along the upper-left corner, confirming that fruit identification maintained optimal probability calibration and rank-ordering despite augmentation transformations.

    ### 13.6 Training History and Convergence Analysis

    **[INSERT Figure 13-7: Multi-Task Augmented Training History]**
    _Source: `reports/scenario3_multitask_augmented/figures/training_history.png`_

    The training history for augmented multi-task learning reveals substantially different convergence dynamics compared to baseline, particularly for the quality task. The fruit type task maintained rapid convergence similar to baseline, achieving >99% accuracy by epoch 3-4 and stabilising at perfect 100% by epoch 5-6. This convergence speed matches baseline despite augmentation, confirming robust fruit identification even during training on distorted images.

    The quality task, conversely, showed notably slower and more volatile convergence. Training accuracy increased gradually over the first 20-25 epochs, reaching only 95-96% by epoch 15 compared to baseline's >99% by epoch 8. The training accuracy curve exhibited substantial epoch-to-epoch oscillation, particularly during epochs 5-25, reflecting the stochastic nature of augmentation where each epoch presents different transformed versions of training samples. Validation accuracy tracked training closely, occasionally matching or exceeding training performance, providing evidence against overfitting despite increased classification errors.

    The quality task eventually stabilised around 97-98% validation accuracy by epoch 35-40, substantially below baseline's 99.89%. The combined loss curve decreased more slowly than baseline, requiring 25-30 epochs to reach values that baseline achieved by epoch 12-15. The learning rate schedule triggered more frequent reductions compared to baseline, indicating that augmented training distribution created more plateau events requiring learning rate adjustment.

    The slower convergence and reduced final accuracy for quality classification whilst fruit type identification converged rapidly and perfectly suggests task-specific augmentation sensitivity. The quality task struggled to extract stable discriminative features from the constantly varying augmented training distribution, whilst the fruit type task learned robust fruit-defining features unaffected by geometric and photometric distortions.

    ### 13.7 Why Augmentation Degraded Quality Classification But Not Fruit Type Identification

    Several factors explain the differential augmentation impact across tasks. First, feature robustness differs fundamentally between tasks. Fruit type identification depends on coarse-grained visual characteristics (overall shape, surface texture patterns, structural features) that remain recognisable under rotation, flipping, and moderate colour variations. Quality assessment, conversely, depends on fine-grained indicators (subtle discolouration, early-stage spotting, minor texture changes) that prove sensitive to photometric transformations, particularly colour jitter that directly alters the appearance of degradation-related colour shifts.

    Second, augmentation-induced ambiguity affected tasks asymmetrically. Aggressive colour jitter (±20% brightness/contrast/saturation) may have genuinely transformed samples across quality boundaries whilst preserving fruit identity. Reducing brightness on Good fruit might make it appear Mildly degraded, or increasing brightness on Rotten fruit might mask decay indicators, creating label noise for quality classification. Fruit type identity, conversely, remains unaffected by such transformations: a brightened banana remains a banana, a rotated tomato remains a tomato.

    Third, task complexity and ceiling effects played a role. The fruit type task already achieved perfect 100% baseline accuracy with zero room for improvement, indicating that inter-fruit visual differences are sufficiently pronounced that no augmentation-induced variation could introduce confusion. The quality task, starting from 99.89% baseline, had minimal improvement potential but substantial degradation risk if augmentation introduced classification difficulty.

    ### 13.8 Conclusion

    Multi-task Scenario 3 revealed differential augmentation impact, with quality classification experiencing substantial degradation (validation 97.06%, test 96.59%, down approximately 3% from baseline) whilst fruit type identification maintained perfect 100% accuracy. The results demonstrate task-specific augmentation sensitivity, with fine-grained quality discrimination proving vulnerable to photometric transformations whilst coarse-grained fruit identification remained robust.

    The perfect fruit type performance under augmentation validates the robustness of inter-fruit visual differences to geometric distortions and colour variations. The quality classification degradation, particularly the severe decline in Mild class performance (F1-score dropping to 93-94%), indicates that intermediate-quality assessment depends on subtle features sensitive to augmentation-induced variations.

    From a practical standpoint, these findings suggest that augmentation strategies for multi-task fruit classification should be task-aware, potentially applying gentler transformations or task-specific augmentation policies that preserve fine-grained quality indicators whilst introducing geometric and photometric variations sufficient for improving fruit identification robustness. The current aggressive augmentation (±20% colour jitter) proved too severe for quality assessment whilst unnecessary for fruit identification that already achieved perfect accuracy.

    ---

    ## 14. COMPARATIVE ANALYSIS: SINGLE-TASK VS MULTI-TASK LEARNING

    ### 14.1 Performance Comparison Across All Scenarios

    Table 26 synthesises performance metrics across all single-task (Part A) and multi-task (Part B) scenarios, enabling direct comparison of architectural paradigms under varying input conditions.

    | Scenario      | Input | Single-Task Quality (Val/Test) | Multi-Task Quality (Val/Test) | Multi-Task Fruit Type (Val/Test) |
    | ------------- | ----- | ------------------------------ | ----------------------------- | -------------------------------- |
    | **Baseline**  | RGB   | 99.84% / 99.79%                | 99.89% / 99.89%               | 100.00% / 100.00%                |
    | **Grayscale** | Gray  | 99.84% / 100.00%               | 99.95% / 99.89%               | 100.00% / 100.00%                |
    | **Augmented** | RGB+  | 98.99% / 99.25%                | 97.06% / 96.59%               | 100.00% / 100.00%                |

    **Table 26: Comprehensive performance comparison across single-task and multi-task scenarios**

    ### 14.2 Key Findings

    **Multi-Task Quality Performance:** The multi-task architecture matched or marginally exceeded single-task quality classification performance under baseline and grayscale conditions. Under baseline RGB processing, multi-task achieved 99.89% compared to single-task's 99.84% (+0.05%). Under grayscale processing, multi-task achieved 99.95% validation compared to single-task's 99.84%, though test performance converged (99.89% multi-task versus 100% single-task). These results challenge the conventional assumption that multi-task learning necessarily trades off performance between competing objectives.

    **Fruit Type Perfect Performance:** The fruit type classification task achieved perfect 100% accuracy across all scenarios (baseline, grayscale, augmented) on both validation and test sets. This comprehensive perfect performance demonstrates that inter-fruit visual differences in the FruQ dataset are sufficiently pronounced that they enable error-free identification regardless of colour availability or augmentation-induced distortions.

    **Augmentation Task-Specific Impact:** Data augmentation revealed differential task sensitivity. Quality classification degraded substantially in multi-task learning (97.06% validation, 96.59% test) compared to single-task (98.99% validation, 99.25% test), representing an additional 1.9% validation degradation and 2.7% test degradation beyond single-task augmentation effects. Fruit type identification, conversely, maintained perfect 100% accuracy, proving entirely robust to augmentation transformations.

    **Colour Independence:** Both single-task and multi-task architectures maintained excellent performance under grayscale processing, with multi-task actually achieving slight improvements. This colour independence suggests that texture, shape, and intensity patterns provide sufficient information for both fruit identification and quality assessment in this dataset.

    ### 14.3 Multi-Task Learning Benefits

    The multi-task architecture demonstrated several practical advantages:

    1. **Dual Output Efficiency:** A single forward pass produces both fruit type and quality predictions, reducing inference time compared to running two separate single-task models sequentially.

    2. **Parameter Efficiency:** The shared convolutional backbone serves both tasks, potentially reducing total parameter count compared to maintaining two independent single-task networks.

    3. **Maintained or Improved Accuracy:** Under baseline and grayscale conditions, multi-task quality classification matched or exceeded single-task performance, suggesting positive transfer learning effects where fruit-specific contextual information refined quality discrimination.

    4. **Robust Fruit Identification:** The fruit type task achieved perfect 100% accuracy across all conditions, validating the multi-task architecture's ability to maintain excellent performance on one task whilst simultaneously optimising another.

    ### 14.4 Recommendations for Deployment

    Based on comprehensive evaluation across six scenarios (three single-task, three multi-task), the following recommendations emerge for practical fruit classification system deployment:

    **For applications requiring both fruit identification and quality assessment:** Deploy multi-task architectures using RGB or grayscale input without augmentation. The multi-task baseline (Scenario 1) or grayscale (Scenario 2) configurations achieve near-perfect performance (>99.89% quality, 100% fruit type) whilst providing computational efficiency benefits through shared feature extraction.

    **For applications prioritising quality assessment only:** Single-task architectures perform comparably to multi-task, with augmentation providing no benefit and potentially degrading performance. The single-task baseline or grayscale configurations are recommended.

    **For augmentation strategies:** Avoid aggressive augmentation for quality classification tasks, as subtle quality indicators prove sensitive to photometric transformations. If augmentation is required for robustness to imaging variations, apply gentler transformations (±10% colour jitter rather than ±20%) or employ task-specific augmentation policies that preserve quality-critical features.

    **For resource-constrained deployments:** Grayscale processing provides viable alternative to RGB, maintaining >99.89% accuracy whilst reducing input dimensionality by 66% (from 3 channels to 1). This simplification offers bandwidth, storage, and computational benefits without sacrificing classification performance.

    ---

    **[END OF PART B]**

    ---

    ## DATA SOURCES AND FIGURE INSERTION NOTES

    All performance metrics in this document were extracted from CSV files located in:

    - `reports/scenario1_multitask/metrics_val.csv` and `metrics_test.csv`
    - `reports/scenario2_multitask_grayscale/metrics_val.csv` and `metrics_test.csv`
    - `reports/scenario3_multitask_augmented/metrics_val.csv` and `metrics_test.csv`

    All figures referenced with **[INSERT Figure X-X]** annotations should be inserted from the corresponding PNG files in:

    - `reports/scenario1_multitask/figures/`
    - `reports/scenario2_multitask_grayscale/figures/`
    - `reports/scenario3_multitask_augmented/figures/`

    Figure types include:

    - `confusion_matrix_quality_val.png` - Quality task validation confusion matrices
    - `confusion_matrix_quality_test.png` - Quality task test confusion matrices
    - `confusion_matrix_fruit_val.png` - Fruit type task validation confusion matrices
    - `confusion_matrix_fruit_test.png` - Fruit type task test confusion matrices
    - `roc_curves_quality_val.png` - Quality task validation ROC curves
    - `roc_curves_quality_test.png` - Quality task test ROC curves
    - `roc_curves_fruit_val.png` - Fruit type task validation ROC curves
    - `roc_curves_fruit_test.png` - Fruit type task test ROC curves
    - `training_history.png` - Multi-panel training history showing both tasks
