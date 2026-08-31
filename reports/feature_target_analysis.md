# Feature-to-Target Analysis Report

This report analyzes features against the target label (0 = Phishing, 1 = Legitimate).

## URLLength
- Unique Values: 439
- Missing: 0
- Correlation w/ Target: -0.2564
- Phishing (0): mean=45.5357, std=54.9294
- Legitimate (1): mean=26.2181, std=4.8118

## DomainLength
- Unique Values: 100
- Missing: 0
- Correlation w/ Target: -0.2808
- Phishing (0): mean=24.3956, std=12.1952
- Legitimate (1): mean=19.2181, std=4.8118

## IsDomainIP :rotating_light: SUSPECT
- **Reasons**: Constant within one class
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: -0.0595
- Phishing (0): mean=0.0061, std=0.0781
- Legitimate (1): mean=0.0000, std=0.0000

## TLD
- Unique Values: 638
- Missing: 0
- Correlation w/ Target: 0.0368
- Phishing (0): mean=313.8149, std=147.4399
- Legitimate (1): mean=324.5872, std=142.3526

## URLSimilarityIndex :rotating_light: SUSPECT
- **Reasons**: Constant within one class, High correlation (0.8598), Flagged by requirements
- Unique Values: 28764
- Missing: 0
- Correlation w/ Target: 0.8598
- Phishing (0): mean=49.7325, std=22.6330
- Legitimate (1): mean=100.0000, std=0.0000

## CharContinuationRate
- Unique Values: 839
- Missing: 0
- Correlation w/ Target: 0.4665
- Phishing (0): mean=0.7286, std=0.2445
- Legitimate (1): mean=0.9329, std=0.1399

## TLDLegitimateProb
- Unique Values: 454
- Missing: 0
- Correlation w/ Target: 0.0946
- Phishing (0): mean=0.2327, std=0.2549
- Legitimate (1): mean=0.2809, std=0.2471

## URLCharProb
- Unique Values: 160504
- Missing: 0
- Correlation w/ Target: 0.4686
- Phishing (0): mean=0.0500, std=0.0116
- Legitimate (1): mean=0.0601, std=0.0072

## TLDLength
- Unique Values: 12
- Missing: 0
- Correlation w/ Target: -0.0782
- Phishing (0): mean=2.8189, std=0.6935
- Legitimate (1): mean=2.7237, std=0.5198

## NoOfSubDomain
- Unique Values: 10
- Missing: 0
- Correlation w/ Target: -0.0080
- Phishing (0): mean=1.1714, std=0.7918
- Legitimate (1): mean=1.1617, std=0.4039

## HasObfuscation :rotating_light: SUSPECT
- **Reasons**: Constant within one class
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: -0.0518
- Phishing (0): mean=0.0047, std=0.0681
- Legitimate (1): mean=0.0000, std=0.0000

## NoOfObfuscatedChar :rotating_light: SUSPECT
- **Reasons**: Constant within one class
- Unique Values: 18
- Missing: 0
- Correlation w/ Target: -0.0153
- Phishing (0): mean=0.0554, std=2.7457
- Legitimate (1): mean=0.0000, std=0.0000

## ObfuscationRatio :rotating_light: SUSPECT
- **Reasons**: Constant within one class
- Unique Values: 127
- Missing: 0
- Correlation w/ Target: -0.0414
- Phishing (0): mean=0.0003, std=0.0057
- Legitimate (1): mean=0.0000, std=0.0000

## NoOfLettersInURL
- Unique Values: 380
- Missing: 0
- Correlation w/ Target: -0.2933
- Phishing (0): mean=27.9538, std=36.7404
- Legitimate (1): mean=12.9206, std=4.7787

## LetterRatioInURL
- Unique Values: 676
- Missing: 0
- Correlation w/ Target: -0.3663
- Phishing (0): mean=0.5676, std=0.1365
- Legitimate (1): mean=0.4765, std=0.0951

## NoOfDegitsInURL
- Unique Values: 174
- Missing: 0
- Correlation w/ Target: -0.1876
- Phishing (0): mean=4.2917, std=16.8236
- Legitimate (1): mean=0.0523, std=0.3563

## DegitRatioInURL
- Unique Values: 556
- Missing: 0
- Correlation w/ Target: -0.4306
- Phishing (0): mean=0.0637, std=0.0962
- Legitimate (1): mean=0.0022, std=0.0148

## NoOfEqualsInURL :rotating_light: SUSPECT
- **Reasons**: Constant within one class
- Unique Values: 24
- Missing: 0
- Correlation w/ Target: -0.0802
- Phishing (0): mean=0.1451, std=1.3676
- Legitimate (1): mean=0.0000, std=0.0000

## NoOfQMarkInURL :rotating_light: SUSPECT
- **Reasons**: Constant within one class
- Unique Values: 5
- Missing: 0
- Correlation w/ Target: -0.1769
- Phishing (0): mean=0.0694, std=0.2926
- Legitimate (1): mean=0.0000, std=0.0000

## NoOfAmpersandInURL :rotating_light: SUSPECT
- **Reasons**: Constant within one class
- Unique Values: 30
- Missing: 0
- Correlation w/ Target: -0.0342
- Phishing (0): mean=0.0582, std=1.2894
- Legitimate (1): mean=0.0000, std=0.0000

## NoOfOtherSpecialCharsInURL
- Unique Values: 66
- Missing: 0
- Correlation w/ Target: -0.3681
- Phishing (0): mean=3.7937, std=4.8457
- Legitimate (1): mean=1.2451, std=0.5029

## SpacialCharRatioInURL
- Unique Values: 229
- Missing: 0
- Correlation w/ Target: -0.5324
- Phishing (0): mean=0.0832, std=0.0357
- Legitimate (1): mean=0.0484, std=0.0190

## IsHTTPS :rotating_light: SUSPECT
- **Reasons**: Constant within one class, Flagged by requirements
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.6141
- Phishing (0): mean=0.4870, std=0.4998
- Legitimate (1): mean=1.0000, std=0.0000

## LineOfCode
- Unique Values: 9448
- Missing: 0
- Correlation w/ Target: 0.2675
- Phishing (0): mean=65.5555, std=203.8092
- Legitimate (1): mean=1944.0201, std=4407.7658

## LargestLineLength
- Unique Values: 22300
- Missing: 0
- Correlation w/ Target: -0.0403
- Phishing (0): mean=20597.9767, std=248710.6668
- Legitimate (1): mean=7346.6115, std=12784.5309

## HasTitle
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.4633
- Phishing (0): mean=0.6745, std=0.4686
- Legitimate (1): mean=0.9988, std=0.0346

## DomainTitleMatchScore :rotating_light: SUSPECT
- **Reasons**: Flagged by requirements
- Unique Values: 147
- Missing: 0
- Correlation w/ Target: 0.5828
- Phishing (0): mean=16.6173, std=36.9195
- Legitimate (1): mean=75.1732, std=42.7370

## URLTitleMatchScore
- Unique Values: 450
- Missing: 0
- Correlation w/ Target: 0.5374
- Phishing (0): mean=21.2616, std=40.5493
- Legitimate (1): mean=75.1732, std=42.7370

## HasFavicon
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.4935
- Phishing (0): mean=0.0873, std=0.2822
- Legitimate (1): mean=0.5674, std=0.4954

## Robots
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.3914
- Phishing (0): mean=0.0669, std=0.2498
- Legitimate (1): mean=0.4178, std=0.4932

## IsResponsive
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.5512
- Phishing (0): mean=0.3136, std=0.4640
- Legitimate (1): mean=0.8536, std=0.3535

## NoOfURLRedirect
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: -0.0478
- Phishing (0): mean=0.1527, std=0.3597
- Legitimate (1): mean=0.1198, std=0.3247

## NoOfSelfRedirect
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: -0.0775
- Phishing (0): mean=0.0581, std=0.2338
- Legitimate (1): mean=0.0272, std=0.1627

## HasDescription
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.6899
- Phishing (0): mean=0.0434, std=0.2038
- Legitimate (1): mean=0.7364, std=0.4406

## NoOfPopup
- Unique Values: 104
- Missing: 0
- Correlation w/ Target: 0.0463
- Phishing (0): mean=0.0090, std=0.1600
- Legitimate (1): mean=0.3917, std=5.3845

## NoOfiFrame
- Unique Values: 104
- Missing: 0
- Correlation w/ Target: 0.2145
- Phishing (0): mean=0.0851, std=0.5290
- Legitimate (1): mean=2.7099, std=7.7779

## HasExternalFormSubmit
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.1673
- Phishing (0): mean=0.0042, std=0.0649
- Legitimate (1): mean=0.0738, std=0.2615

## HasSocialNet
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.7828
- Phishing (0): mean=0.0052, std=0.0722
- Legitimate (1): mean=0.7943, std=0.4042

## HasSubmitButton
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.5791
- Phishing (0): mean=0.0844, std=0.2780
- Legitimate (1): mean=0.6619, std=0.4731

## HasHiddenFields
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.5072
- Phishing (0): mean=0.0932, std=0.2907
- Legitimate (1): mean=0.5910, std=0.4916

## HasPasswordField
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.1394
- Phishing (0): mean=0.0531, std=0.2242
- Legitimate (1): mean=0.1385, std=0.3454

## Bank
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.1898
- Phishing (0): mean=0.0535, std=0.2251
- Legitimate (1): mean=0.1814, std=0.3853

## Pay
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.3587
- Phishing (0): mean=0.0607, std=0.2388
- Legitimate (1): mean=0.3698, std=0.4828

## Crypto
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.0993
- Phishing (0): mean=0.0060, std=0.0770
- Legitimate (1): mean=0.0363, std=0.1871

## HasCopyrightInfo
- Unique Values: 2
- Missing: 0
- Correlation w/ Target: 0.7428
- Phishing (0): mean=0.0573, std=0.2324
- Legitimate (1): mean=0.8084, std=0.3936

## NoOfImage
- Unique Values: 852
- Missing: 0
- Correlation w/ Target: 0.2820
- Phishing (0): mean=0.8716, std=3.2806
- Legitimate (1): mean=44.6060, std=96.9667

## NoOfCSS
- Unique Values: 188
- Missing: 0
- Correlation w/ Target: 0.0576
- Phishing (0): mean=0.4405, std=1.4388
- Legitimate (1): mean=10.8535, std=117.6646

## NoOfJS
- Unique Values: 233
- Missing: 0
- Correlation w/ Target: 0.3412
- Phishing (0): mean=0.8956, std=3.3960
- Legitimate (1): mean=17.7562, std=30.1477

## NoOfSelfRef
- Unique Values: 1226
- Missing: 0
- Correlation w/ Target: 0.3481
- Phishing (0): mean=0.5034, std=3.2167
- Legitimate (1): mean=112.9781, std=197.4693

## NoOfEmptyRef
- Unique Values: 264
- Missing: 0
- Correlation w/ Target: 0.1018
- Phishing (0): mean=0.1534, std=1.8633
- Legitimate (1): mean=4.0757, std=24.9426

## NoOfExternalRef
- Unique Values: 1050
- Missing: 0
- Correlation w/ Target: 0.2888
- Phishing (0): mean=1.1184, std=3.0467
- Legitimate (1): mean=85.0050, std=181.2721
