\# Intelligent Media Processing Pipeline



A backend system for asynchronous image upload and intelligent media analysis.



The system accepts image uploads, stores metadata, processes images asynchronously, and generates structured analysis results.



\## Features



\- Image upload API

\- Unique processing ID for every upload

\- SQLite database persistence

\- Asynchronous background processing

\- Processing status tracking

\- Blur detection

\- Brightness analysis

\- Image dimension validation

\- SHA-256 file hash generation

\- Perceptual hash generation

\- Duplicate image detection

\- Indian vehicle number format validation structure

\- Screenshot/photo-of-photo heuristics

\- Failure handling

\- Structured JSON analysis results



\## Architecture



```text

Client

&#x20; |

&#x20; | POST /api/v1/images

&#x20; v

FastAPI API

&#x20; |

&#x20; | 1. Validate image

&#x20; | 2. Save image locally

&#x20; | 3. Store metadata in SQLite

&#x20; | 4. Create processing ID

&#x20; |

&#x20; v

Background Worker Thread

&#x20; |

&#x20; | status = processing

&#x20; |

&#x20; +--> Blur Detection

&#x20; +--> Brightness Analysis

&#x20; +--> Dimension Validation

&#x20; +--> SHA-256 Hash

&#x20; +--> Perceptual Hash

&#x20; +--> Duplicate Detection

&#x20; +--> Vehicle Number Validation

&#x20; +--> Screenshot Heuristics

&#x20; |

&#x20; v

SQLite Database

&#x20; |

&#x20; v

Results API

