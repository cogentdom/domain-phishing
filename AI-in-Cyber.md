# AI in Cybersecurity: Detecting Domain Phishing
 
### Slide 1: Title Slide
• Title: AI in Cybersecurity: Detecting Domain Phishing  
• Subtitle: Leveraging Deep Learning Models  
• Author: Dominik Huffield  

### Slide 2: Dominik Huffield
• Involved in 5 different startups  
• Big Tech – ½ trillion valuation  
• Industries: material science, e-commerce, cyber security, medical  
• 2x Founder  
• Degree in Economics  

### Slide 3: Introductory stuff
• AI Development Tiers  
    o Tier 0: Mathematician, actual designers of model  
    o Tier 1: Data Scientist, utilizes raw model by training over data  
    o Tier 2: AI/ML Engineer, Deploys trained model to infostructure  
    o Tier 3: End User, utilizes tool to augment their workflow  
• Economists
    o First Practitioners of Data Science  
    o High Frequency Trading price ceiling  
• Hackers using AI  
    o A finance worker at a multinational firm was tricked into paying out $25 million to fraudsters using deepfake technology  
    o Hong Kong February 2024  
    o Social Engineering  
• Data Surpasses Oil as most valuable commodity  
    o May 6th 2017  
 
### Slide 4: A History of AI
• Comparison of traditional cybersecurity methods vs. AI-driven approaches  
• In 1988, Deep Thought, Deep Blue’s predecessor, beat grandmaster Bent Larsen  
    o the system was easily defeated twice the following year  
• 1997 deep blue vs Garry Kasparov 3-1(3 draws, 6 games)  
    o Built using a Bayesian structure  
    o The system searched through 100 and 200 million positions per second  
    o Kasparov says a good human plus a machine is the best combination  
• 2011 Watson competed on jeopardy and won against two all-time champions  
• 2016 Alpha Go defeats world champion 4-1  
    o Built using neural networks  
• Then, in 2017, AlphaGo defeats Ke Jie 3-0, the worlds number 1 rated player  
• Deep Blue programmed by professional Chess players  
• Alpha Go trained using 100,000 professional Go matches  
• Defeats world champion Lee Sedol 4-1  
 
### Slide 5: What is AI
• Definition of machine learning  
    o Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data and thus perform tasks without explicit instructions. Recently, artificial neural networks have been able to surpass many previous approaches in performance.  
• Definition of deep learning  
    o Deep learning is the subset of machine learning methods based on neural networks with representation learning. The adjective "deep" refers to the use of multiple layers in the network. Methods used can be either supervised, semi-supervised or unsupervised.  
• Basics of neural networks  
    o RNN  
        ▪ Input -> Computation -> Output  
        ▪ Uses output of previous step as input in addition to next step  
    o LSTM  
        ▪ Solves long term dependency problem  
        ▪ Vanishing gradient  
        ▪ Adds internal state  
    o Sequence processing: forecasting, text processing  
    o CNN  
        ▪ Hidden Layers Convolutions  
        ▪ Image processing  
        ▪ Filters detect patterns to engineer features  
• Relevance of deep learning in cybersecurity  
 
 
### Slide 6: Introduction to Cybersecurity
• Brief overview of cybersecurity threats  
    o Cybersecurity aims to protect individuals’ and organizations’ systems, applications, computing devices, sensitive data and financial assets against computer viruses, sophisticated and costly ransomware attacks, and more.  
• Importance of proactive measures  
• Introduction to domain phishing  
    o is a counterfeit website or portal designed by cybercriminals to trick users into believing that they are interacting with a legitimate site.  
    o To deceitfully acquire sensitive information such as login credentials, credit card details, or other personal data that can be exploited for fraudulent purposes.  
• Types of Attacks  
    o Reconnaissance: Techniques that actively or passively gather information to plan future targeted attacks.  
    o Resource development: Involves attackers purchasing or stealing resources to use them for a future attack.  
    o Initial access: Techniques where adversaries try to gain a foothold in your network through different attack vectors.  
    o Execution: Adversary techniques that try to run malicious code on a local or remote system.  
    o Persistence: Tactics that involve adversaries trying to maintain their foothold in your local or remote network.  
    o Privilege escalation: When an adversary tries to gain higher-level permission into your organization’s network.  
    o Defense evasion: Adversary techniques to avoid detection when they move through your network.  
    o Credential access: Tactics focused on retrieving sensitive credentials such as passwords.  
    o Discovery: When adversaries try to gain an understanding of how your systems work.  
    o Lateral movement: Involves adversaries that enter and control systems, moving through your network.  
    o Collection: Techniques that gather information from relevant sources within your organization.  
    o Command and Control (C2 or C&C): When adversaries communicate with compromised systems to gain control.  
    o Exfiltration: Consists of techniques that straight up steal data from your network.  
    o Impact: When adversaries focus on disrupting data availability or integrity and interrupting business operations.  
 
### Slide 7: What is Domain Phishing?
• Definition of domain phishing  
    o the most common type of attack is phishing, which is a type of attack that aims to steal or misuse a user’s personal information, including account information, identity, passwords, and credit card details.  
    o
• Common tactics used by attackers  
• Impact on organizations and individuals  
 
### Slide 8: Traditional Approaches vs. AI in Cybersecurity
• Comparison of traditional cybersecurity methods vs. AI-driven approaches  
    o Reactive vs proactive  
    o Blindly adding preventative measures  
• Benefits of using AI and deep learning in cybersecurity  
    o Network wide behavior monitoring  
    o Automated flagging of malicious actors  
    o Automated remediation of compromised devices  
 
### Slide 9: Data Science Methodologies
• Regression vs Classification vs Clustering  
• Supervised vs Unsupervised  
• Data Collection  
• Research & Development  
    o Training  
    o Embeddings  
    o Architecture  
• Deploying and Serving  
    o Sagemaker  
    o Productionize  
• Contiguous Learning  
    o Closed loop  
    o Federated learning arm  

### Slide 10: Dataset Preparation, Preprocessing, Feature Engineering
• Importance of high-quality data  
• Types of features to consider  
    o Domain  
    o Sub-domain  
    o Suffix  
    o URL Length  
    o Path Length  
    o SSL Certified  
    o Num Dots  
    o Num Dashes    
    o Num Dash in Hostname  
    o Num Under Scores  
    o Who is URL extraction  
    o Origin  
    o DNS Server  
    o IP address?  
• Data collection and preprocessing steps  
    o Shape (rows, 256, 69)  

### Slide 11: Building the Deep Learning Model
• Architecture overview (e.g., convolutional neural network, recurrent neural network)  
• Training process  
• Evaluation metrics (e.g., accuracy, precision, recall)  

### Slide 12: Model Evaluation and Validation
• Cross-validation techniques  
• Testing the model with unseen data  
• Fine-tuning and optimization strategies  

### Slide 13: Challenges and Future Directions
• Addressing imbalanced datasets  
    o Deterministic vs probabilistic  
        ▪ Learns only within domain  
• Adapting to evolving phishing techniques  
• Integration with existing cybersecurity systems  

### Slide 14: My thoughts on AI
• Breadth + Depth = a plane that tech calls “intelligence”  
• Breadth is human domain (Prosthetic hand allegory)  
• Don’t portray AI as humanoid  
• Layers of Identity  
    o Layer 1: Neurology  
    o Layer 2: Psychiatry  
    o Layer 3: Psychology  
    o Layer 4: Artificial  
 
### Slide 15: Learn to Work with data yourself
• Encouragement for further research and development  

### Slide 16: References and Q&A
• Open floor for questions and discussion  
-