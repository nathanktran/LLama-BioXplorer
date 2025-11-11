// filename -App.js

import React from "react";
import "./App.css";
import Navbar from "./components/Navbar";
import {
    BrowserRouter as Router,
    Routes,
    Route,
} from "react-router-dom";
import Home from "./pages";
//import About from "./pages/about";
import Events from "./pages/search";
import AnnualReport from "./pages/upload";
import Download from "./pages/download";
import BioGPT from "./pages/biogpt";
// import Teams from "./pages/team";
import Blogs from "./pages/contactUs";
// import SignUp from "./pages/signup";

function App() {
    return (
        <Router>
            <Navbar />
            <Routes>
                <Route path="/" element={<Home />} />
                <Route
                    path="/search"
                    element={<Events />}
                />
                <Route
                    path="/upload"
                    element={<AnnualReport />}
                />
                <Route path="/download" element={<Download />} />
                <Route path="/biogpt" element={<BioGPT />} />
                <Route path="/contactUs" element={<Blogs />} />
                {/* <Route
                    path="/sign-up"
                    element={<SignUp />}
                /> */}
            </Routes>
        </Router>
    );
}

export default App;
